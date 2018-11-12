import numpy as np

# from rllab.envs.normalized_env import normalize
from maci.misc import tf_utils
from maci.learners import MADDPG
from maci.misc.kernel import adaptive_isotropic_gaussian_kernel
from maci.replay_buffers import SimpleReplayBuffer
from maci.value_functions.sq_value_function import NNQFunction
from maci.misc.sga import SymplecticOptimizer
from maci.policies import DeterministicNNPolicy
from maci.misc.sampler import MASampler
from maci.environments import DifferentialGame
from rllab.misc import logger
import gtimer as gt
import datetime
from copy import deepcopy
import tensorflow as tf

import maci.misc.tf_utils as U
import os

def main():
    joint = False
    opponent_modelling = False
    SGA = True

    game_name = 'ma_softq'
    model_name = 'MADDPG'
    if not joint:
        model_name = 'DDPG'
        if SGA:
            model_name = 'DDPG_SGA'
    if joint and opponent_modelling:
        model_name = 'DDPG-OM'
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f %Z')

    suffix = '{}/{}/{}'.format(game_name, model_name, timestamp)

    agent_num = 2
    u_range = 10.

    logger.add_tabular_output('./log/{}.csv'.format(suffix))
    snapshot_dir = './snapshot/{}'.format(suffix)
    policy_dir = './policy/{}'.format(suffix)
    os.makedirs(snapshot_dir, exist_ok=True)
    os.makedirs(policy_dir, exist_ok=True)
    logger.set_snapshot_dir(snapshot_dir)

    # policy_file = open('{}/policy.csv'.format(policy_dir), 'a')
    env = DifferentialGame(game_name=game_name, agent_num=agent_num)
    agents = []
    M = 100
    batch_size = 64
    sampler = MASampler(agent_num=agent_num, joint=joint, max_path_length=30, min_pool_size=100, batch_size=batch_size)

    base_kwargs = {
        'sampler': sampler,
        'epoch_length': 1,
        'n_epochs': 16000,
        'n_train_repeat': 1,
        'eval_render': True,
        'eval_n_episodes': 10
    }

    with U.single_threaded_session():
        for i in range(agent_num):
            pool = SimpleReplayBuffer(env.env_specs, max_replay_buffer_size=1e6, joint=joint, agent_id=i)
            policy = DeterministicNNPolicy(env.env_specs,
                                           hidden_layer_sizes=(M, M),
                                           squash=True, u_range=u_range, joint=False,
                                           agent_id=i)
            target_policy = DeterministicNNPolicy(env.env_specs,
                                                  hidden_layer_sizes=(M, M),
                                                  name='target_policy',
                                                  squash=True, u_range=u_range, joint=False,
                                                  agent_id=i)
            opponent_policy = None
            if opponent_modelling:
                opponent_policy = DeterministicNNPolicy(env.env_specs,
                                               hidden_layer_sizes=(M, M),
                                               name='opponent_policy',
                                               squash=True, u_range=u_range, joint=False,
                                               opponent_policy=True,
                                               agent_id=i)
            qf = NNQFunction(env_spec=env.env_specs, hidden_layer_sizes=[M, M], joint=joint, agent_id=i)
            target_qf = NNQFunction(env_spec=env.env_specs, hidden_layer_sizes=[M, M], name='target_qf', joint=joint, agent_id=i)
            plotter = None


            agent = MADDPG(
                base_kwargs=base_kwargs,
                agent_id=i,
                env=env,
                pool=pool,
                qf=qf,
                target_qf=target_qf,
                policy=policy,
                target_policy=target_policy,
                opponent_policy=opponent_policy,
                plotter=plotter,
                policy_lr=3e-4,
                qf_lr=3e-4,
                joint=joint,
                opponent_modelling=opponent_modelling,
                td_target_update_interval=10,
                discount=0.99,
                reward_scale=0.1,
                save_full_state=False,
                SGA=SGA)

            agents.append(agent)

        sampler.initialize(env, agents)

        sga_train_ops = []
        sga_sess = None
        if SGA:
            with tf.variable_scope('sga_policy_opt', reuse=tf.AUTO_REUSE):
                a0_vars = agents[0].policy.get_params_internal()
                a1_vars = agents[1].policy.get_params_internal()
                xs = a0_vars + a1_vars
                a0_grad = tf.gradients(agents[0]._pg_loss, a0_vars)
                a1_grad = tf.gradients(agents[1]._pg_loss, a1_vars)
                Xi = a0_grad + a1_grad
                apply_vec = list(zip(Xi,xs))
                optimizer = SymplecticOptimizer(learning_rate=agents[0]._qf_lr)
                with tf.control_dependencies([g for (g, v) in apply_vec]):
                    sga_train_ops.append(optimizer.apply_gradients(apply_vec))

                a0_vars = agents[0].qf.get_params_internal()
                a1_vars = agents[1].qf.get_params_internal()
                xs = a0_vars + a1_vars
                a0_grad = tf.gradients(agents[0]._bellman_residual, a0_vars)
                a1_grad = tf.gradients(agents[1]._bellman_residual, a1_vars)
                Xi = a0_grad + a1_grad
                apply_vec = list(zip(Xi,xs))

                optimizer = SymplecticOptimizer(learning_rate=agents[0]._qf_lr)
                with tf.control_dependencies([g for (g, v) in apply_vec]):
                    sga_train_ops.append(optimizer.apply_gradients(apply_vec))
            sga_sess = tf_utils.get_default_session()
            sga_sess.run(tf.global_variables_initializer())

        def do_sga_training(iteration, feed_dict):
            sga_sess.run(sga_train_ops, feed_dict)
            if iteration % agents[0]._qf_target_update_interval == 0:
                for agent in agents:
                    sga_sess.run(agent._target_ops)

        for agent in agents:
            agent._init_training()
        gt.rename_root('MARLAlgorithm')
        gt.reset()
        gt.set_def_unique(False)
        initial_exploration_done = False
        noise = .1


        for agent in agents:
            agent.policy.set_noise_level(noise)

        for epoch in gt.timed_for(range(base_kwargs['n_epochs'] + 1)):
            logger.push_prefix('Epoch #%d | ' % epoch)
            for t in range(base_kwargs['epoch_length']):
                # TODO.code consolidation: Add control interval to sampler
                if not initial_exploration_done:
                    if epoch >= 1000:
                        initial_exploration_done = True
                sampler.sample()
                print('Sampling')
                if not initial_exploration_done:
                    continue
                gt.stamp('sample')
                print('Sample Done')
                if epoch == base_kwargs['n_epochs']:
                    noise = 0.01
                    for agent in agents:
                        agent.policy.set_noise_level(noise)
                    # alpha = .1
                if epoch > base_kwargs['n_epochs'] / 10:
                    noise = 0.01
                    for agent in agents:
                        agent.policy.set_noise_level(noise)
                    # alpha = .1
                if epoch > base_kwargs['n_epochs'] / 5:
                    noise = 0.01
                    for agent in agents:
                        agent.policy.set_noise_level(noise)
                if epoch > base_kwargs['n_epochs'] / 6:
                    noise = 0.001
                    for agent in agents:
                        agent.policy.set_noise_level(noise)



                for j in range(base_kwargs['n_train_repeat']):
                    batch_n = []
                    recent_batch_n = []
                    indices = None
                    sga_feed_dict = {}
                    for i, agent in enumerate(agents):
                        if i == 0:
                            batch = agent.pool.random_batch(batch_size)
                            indices = agent.pool.indices
                            if opponent_modelling:
                                receent_indices = list(range(agent.pool._top-batch_size, agent.pool._top))
                        batch_n.append(agent.pool.random_batch_by_indices(indices))
                        if opponent_modelling:
                            recent_batch_n.append(agent.pool.random_batch_by_indices(receent_indices))

                    print(len(batch_n))
                    target_next_actions_n = np.array([agent._target_policy.get_actions(batch['next_observations']) for agent, batch in zip(agents, batch_n)])
                    opponent_actions_n = np.array([batch['actions'] for batch in batch_n])
                    recent_opponent_actions_n = np.array([batch['actions'] for batch in recent_batch_n])
                    recent_opponent_observations_n = np.array([batch['observations'] for batch in recent_batch_n])
                    print('=====target====behaviour')
                    print(agents[0]._target_policy.get_actions(batch_n[0]['next_observations'])[0])
                    a1 = agents[0]._policy.get_actions(batch_n[0]['next_observations'])[0][0]
                    a2 = agents[1]._policy.get_actions(batch_n[1]['next_observations'])[0][0]
                    print(a1, a2)
                    with open('{}/policy.csv'.format(policy_dir), 'a') as f:
                        f.write('{},{}\n'.format(a1, a2))
                    print('============')
                    for i, agent in enumerate(agents):
                        batch_n[i]['next_actions'] = deepcopy(target_next_actions_n[i])
                        batch_n[i]['opponent_actions'] = np.reshape(np.delete(deepcopy(opponent_actions_n), i, 0), (-1, agent._opponent_action_dim))
                        if agent.joint:
                            if agent.opponent_modelling:
                                batch_n[i]['recent_opponent_observations'] = recent_opponent_observations_n[i]
                                batch_n[i]['recent_opponent_actions'] = np.reshape(np.delete(deepcopy(recent_opponent_actions_n), i, 0), (-1, agent._opponent_action_dim))
                                batch_n[i]['opponent_next_actions'] = agent.opponent_policy.get_actions(batch_n[i]['next_observations'])
                            else:
                                batch_n[i]['opponent_next_actions'] = np.reshape(np.delete(deepcopy(target_next_actions_n), i, 0), (-1, agent._opponent_action_dim))
                        if not SGA:
                            agent._do_training(iteration=t + epoch * agent._epoch_length, batch=batch_n[i])
                        else:
                            sga_feed_dict.update(agent._get_feed_dict(batch_n[i]))
                    if SGA:
                        do_sga_training(iteration=t + epoch * agent._epoch_length, feed_dict=sga_feed_dict)
                gt.stamp('train')

            # self._evaluate(epoch)

            # for agent in agents:
            #     params = agent.get_snapshot(epoch)
            #     logger.save_itr_params(epoch, params)
            # times_itrs = gt.get_times().stamps.itrs
            #
            # eval_time = times_itrs['eval'][-1] if epoch > 1 else 0
            # total_time = gt.get_times().total
            # logger.record_tabular('time-train', times_itrs['train'][-1])
            # logger.record_tabular('time-eval', eval_time)
            # logger.record_tabular('time-sample', times_itrs['sample'][-1])
            # logger.record_tabular('time-total', total_time)
            # logger.record_tabular('epoch', epoch)

            # sampler.log_diagnostics()

            # logger.dump_tabular(with_prefix=False)
            logger.pop_prefix()
            sampler.terminate()


if __name__ == '__main__':
    main()