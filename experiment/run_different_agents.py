import numpy as np


from maci.learners import MADDPG, MAVBAC, MASQL
from maci.misc.kernel import adaptive_isotropic_gaussian_kernel
from maci.replay_buffers import SimpleReplayBuffer
from maci.value_functions.sq_value_function import NNQFunction, NNJointQFunction
from maci.misc.plotter import QFPolicyPlotter
from maci.policies import DeterministicNNPolicy, StochasticNNConditionalPolicy, StochasticNNPolicy
from maci.misc.sampler import MASampler
from maci.environments import DifferentialGame
from rllab.misc import logger
import gtimer as gt
import datetime
from copy import deepcopy

import maci.misc.tf_utils as U
import os


def masql_agent(model_name, i, env, M, u_range, base_kwargs):
    joint = True
    pool = SimpleReplayBuffer(env.env_specs, max_replay_buffer_size=1e6, joint=joint, agent_id=i)
    policy = StochasticNNPolicy(env.env_specs,
                                hidden_layer_sizes=(M, M),
                                squash=True, u_range=10., joint=joint,
                                agent_id=i)

    qf = NNQFunction(env_spec=env.env_specs, hidden_layer_sizes=[M, M], joint=joint, agent_id=i)
    target_qf = NNQFunction(env_spec=env.env_specs, hidden_layer_sizes=[M, M], name='target_qf', joint=joint,
                            agent_id=i)

    plotter = QFPolicyPlotter(
        qf=qf,
        policy=policy,
        obs_lst=np.array([[0]]),
        default_action=[np.nan, np.nan],
        n_samples=100,
        agent_id=i)

    agent = MASQL(
        base_kwargs=base_kwargs,
        agent_id=i,
        env=env,
        pool=pool,
        qf=qf,
        target_qf=target_qf,
        policy=policy,
        plotter=plotter,
        policy_lr=3e-4,
        qf_lr=3e-4,
        tau=0.01,
        value_n_particles=16,
        td_target_update_interval=10,
        kernel_fn=adaptive_isotropic_gaussian_kernel,
        kernel_n_particles=32,
        kernel_update_ratio=0.5,
        discount=0.99,
        reward_scale=1,
        save_full_state=False)
    return agent

def pr2ac_agent(model_name, i, env, M, u_range, base_kwargs):
    joint = False
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
    conditional_policy = StochasticNNConditionalPolicy(env.env_specs,
                                                       hidden_layer_sizes=(M, M),
                                                       name='conditional_policy',
                                                       squash=True, u_range=u_range, joint=False,
                                                       agent_id=i)

    joint_qf = NNJointQFunction(env_spec=env.env_specs, hidden_layer_sizes=[M, M], joint=joint, agent_id=i)
    target_joint_qf = NNJointQFunction(env_spec=env.env_specs, hidden_layer_sizes=[M, M], name='target_joint_qf',
                                       joint=joint, agent_id=i)

    qf = NNQFunction(env_spec=env.env_specs, hidden_layer_sizes=[M, M], joint=False, agent_id=i)
    plotter = None

    agent = MAVBAC(
        base_kwargs=base_kwargs,
        agent_id=i,
        env=env,
        pool=pool,
        joint_qf=joint_qf,
        target_joint_qf=target_joint_qf,
        qf=qf,
        policy=policy,
        target_policy=target_policy,
        conditional_policy=conditional_policy,
        plotter=plotter,
        policy_lr=3e-4,
        qf_lr=3e-4,
        joint=False,
        value_n_particles=16,
        kernel_fn=adaptive_isotropic_gaussian_kernel,
        kernel_n_particles=32,
        kernel_update_ratio=0.5,
        td_target_update_interval=5,
        discount=0.99,
        reward_scale=1,
        tau=0.01,
        save_full_state=False)
    return agent


def ddpg_agent(joint, opponent_modelling, model_name, i, env, M, u_range, base_kwargs):
    # joint = True
    # opponent_modelling = False
    print(model_name)

    print(joint, opponent_modelling)
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
    target_qf = NNQFunction(env_spec=env.env_specs, hidden_layer_sizes=[M, M], name='target_qf', joint=joint,
                            agent_id=i)
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
        save_full_state=False)

    return agent



def main():
    game_name = 'ma_softq'
    model_names = ['PR2AC', 'MASQL']
    model_name = '_'.join(model_names)

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
    sampler = MASampler(agent_num=agent_num, joint=True, max_path_length=30, min_pool_size=100, batch_size=batch_size)

    base_kwargs = {
        'sampler': sampler,
        'epoch_length': 1,
        'n_epochs': 16000,
        'n_train_repeat': 1,
        'eval_render': True,
        'eval_n_episodes': 10
    }

    with U.single_threaded_session():
        for i, model_name in enumerate(model_names):
            if model_name == 'PR2AC':
                agent = pr2ac_agent(model_name, i, env, M, u_range, base_kwargs)
            elif model_name == 'MASQL':
                agent = masql_agent(model_name, i, env, M, u_range, base_kwargs)
            else:
                if model_name == 'DDPG':
                    joint = False
                    opponent_modelling = False
                elif model_name == 'MADDPG':
                    joint = True
                    opponent_modelling = False
                elif model_name == 'DDPG-OM':
                    joint = True
                    opponent_modelling = True
                agent = ddpg_agent(joint, opponent_modelling, model_names, i, env, M, u_range, base_kwargs)

            agents.append(agent)

        sampler.initialize(env, agents)

        for agent in agents:
            agent._init_training()
        gt.rename_root('MARLAlgorithm')
        gt.reset()
        gt.set_def_unique(False)
        initial_exploration_done = False
        # noise = .1
        noise = 1.
        alpha = .5


        for agent in agents:
            try:
                agent.policy.set_noise_level(noise)
            except:
                pass

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
                        try:
                            agent.policy.set_noise_level(noise)
                        except:
                            pass
                    # alpha = .1
                if epoch > base_kwargs['n_epochs'] / 10:
                    noise = 0.01
                    for agent in agents:
                        try:
                            agent.policy.set_noise_level(noise)
                        except:
                            pass
                    # alpha = .1
                if epoch > base_kwargs['n_epochs'] / 5:
                    noise = 0.01
                    for agent in agents:
                        try:
                            agent.policy.set_noise_level(noise)
                        except:
                            pass
                if epoch > base_kwargs['n_epochs'] / 6:
                    noise = 0.001
                    for agent in agents:
                        try:
                            agent.policy.set_noise_level(noise)
                        except:
                            pass

                for j in range(base_kwargs['n_train_repeat']):
                    batch_n = []
                    recent_batch_n = []
                    indices = None
                    for i, agent in enumerate(agents):
                        if i == 0:
                            batch = agent.pool.random_batch(batch_size)
                            indices = agent.pool.indices
                            receent_indices = list(range(agent.pool._top-batch_size, agent.pool._top))

                        batch_n.append(agent.pool.random_batch_by_indices(indices))

                        recent_batch_n.append(agent.pool.random_batch_by_indices(receent_indices))

                    print(len(batch_n))
                    target_next_actions_n = []
                    try:
                        for agent, batch in zip(agents, batch_n):
                            target_next_actions_n.append(agent._target_policy.get_actions(batch['next_observations']))
                    except:
                        pass
                    # next_actions_n = np.array([agent._policy.get_actions(batch['next_observations']) for agent, batch in zip(agents, batch_n)])
                    opponent_actions_n = np.array([batch['actions'] for batch in batch_n])
                    recent_opponent_actions_n = np.array([batch['actions'] for batch in recent_batch_n])
                    recent_opponent_observations_n = np.array([batch['observations'] for batch in recent_batch_n])
                    print('=====target====behaviour')
                    # print(agents[0]._target_policy.get_actions(batch_n[0]['next_observations'])[0])
                    a1 = agents[0]._policy.get_actions(batch_n[0]['next_observations'])[0][0]
                    a2 = agents[1]._policy.get_actions(batch_n[1]['next_observations'])[0][0]
                    print(a1, a2)

                    for i, agent in enumerate(agents):
                        try:
                            batch_n[i]['next_actions'] = deepcopy(target_next_actions_n[i])
                        except:
                            pass
                        batch_n[i]['opponent_actions'] = np.reshape(np.delete(deepcopy(opponent_actions_n), i, 0), (-1, agent._opponent_action_dim))
                        if agent.joint:
                            if agent.opponent_modelling:
                                batch_n[i]['recent_opponent_observations'] = recent_opponent_observations_n[i]
                                batch_n[i]['recent_opponent_actions'] = np.reshape(np.delete(deepcopy(recent_opponent_actions_n), i, 0), (-1, agent._opponent_action_dim))
                                batch_n[i]['opponent_next_actions'] = agent.opponent_policy.get_actions(batch_n[i]['next_observations'])
                            else:
                                batch_n[i]['opponent_next_actions'] = np.reshape(np.delete(deepcopy(target_next_actions_n), i, 0), (-1, agent._opponent_action_dim))
                        if isinstance(agent, MAVBAC) or isinstance(agent, MASQL):

                            agent._do_training(iteration=t + epoch * agent._epoch_length, batch=batch_n[i], annealing=alpha)
                        else:
                            agent._do_training(iteration=t + epoch * agent._epoch_length, batch=batch_n[i])
                gt.stamp('train')

            logger.pop_prefix()
            sampler.terminate()


if __name__ == '__main__':
    main()