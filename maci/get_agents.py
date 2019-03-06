import numpy as np
import tensorflow as tf
from maci.learners import MADDPG, MAVBAC, MASQL, ROMMEO
from maci.misc.kernel import adaptive_isotropic_gaussian_kernel
from maci.replay_buffers import SimpleReplayBuffer
from maci.value_functions.sq_value_function import NNQFunction, NNJointQFunction
from maci.policies import StochasticNNConditionalPolicy, StochasticNNPolicy
from maci.policies.deterministic_policy import DeterministicNNPolicy, ConditionalDeterministicNNPolicy
from maci.policies.uniform_policy import UniformPolicy
from maci.policies.level_k_policy import MultiLevelPolicy, GeneralizedMultiLevelPolicy
from maci.policies.gaussian_policy import GaussianConditionalPolicy, GaussianPolicy

def masql_agent(model_name, i, env, M, u_range, base_kwargs, game_name='matrix'):
    joint = True
    squash = True
    squash_func = tf.tanh
    sampling = False
    if 'particle' in game_name:
        sampling = True
        squash_func = tf.nn.softmax

    pool = SimpleReplayBuffer(env.env_specs, max_replay_buffer_size=1e6, joint=joint, agent_id=i)
    policy = StochasticNNPolicy(env.env_specs,
                                hidden_layer_sizes=(M, M),
                                squash=squash, squash_func=squash_func, sampling=sampling, u_range=u_range, joint=joint,
                                agent_id=i)

    qf = NNQFunction(env_spec=env.env_specs, hidden_layer_sizes=[M, M], joint=joint, agent_id=i)
    target_qf = NNQFunction(env_spec=env.env_specs, hidden_layer_sizes=[M, M], name='target_qf', joint=joint,
                            agent_id=i)

    plotter = None

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


def get_level_k_policy(env, k, M, agent_id, u_range, opponent_conditional_policy, game_name='matrix'):
    urange = [-1, 1.]
    if_softmax = False
    if 'particle' in game_name:
        urange = [-100., 100.]
        if_softmax = True
    squash = True
    squash_func = tf.tanh
    correct_tanh = True

    sampling = False
    if 'particle' in game_name:
        sampling = True
        squash_func = tf.nn.softmax
        correct_tanh = False
    # print('env spec', env.env_specs)
    opponent = False
    if k % 2 == 1:
        opponent = True

    base_policy = UniformPolicy(env.env_specs, agent_id=agent_id, opponent=opponent, urange=urange, if_softmax=if_softmax)
    conditional_policy = ConditionalDeterministicNNPolicy(env.env_specs,
                                                          hidden_layer_sizes=(M, M),
                                                          name='conditional_policy',
                                                          squash=squash, squash_func=squash_func, sampling=sampling, u_range=u_range, joint=False,
                                                          agent_id=agent_id)
    k_policy = MultiLevelPolicy(env_spec=env.env_specs,
                                k=k,
                                base_policy=base_policy,
                                conditional_policy=conditional_policy,
                                opponent_conditional_policy=opponent_conditional_policy,
                                agent_id=agent_id)
    with tf.variable_scope('target_levelk_{}'.format(agent_id), reuse=True):
        target_k_policy = MultiLevelPolicy(env_spec=env.env_specs,
                                    k=k,
                                    base_policy=base_policy,
                                    conditional_policy=conditional_policy,
                                    opponent_conditional_policy=opponent_conditional_policy,
                                    agent_id=agent_id)
    return k_policy, target_k_policy


def pr2ac_agent(model_name, i, env, M, u_range, base_kwargs, k=0, g=False, mu=1.5, game_name='matrix', aux=True):
    joint = False
    squash = True
    squash_func = tf.tanh
    correct_tanh = True
    sampling = False
    if 'particle' in game_name:
        sampling = True
        squash = True
        squash_func = tf.nn.softmax
        correct_tanh = False

    pool = SimpleReplayBuffer(env.env_specs, max_replay_buffer_size=1e6, joint=joint, agent_id=i)

    opponent_conditional_policy = StochasticNNConditionalPolicy(env.env_specs,
                                                       hidden_layer_sizes=(M, M),
                                                       name='opponent_conditional_policy',
                                                       squash=squash, squash_func=squash_func,sampling=sampling, u_range=u_range, joint=joint,
                                                       agent_id=i)


    if g:
        policies = []
        target_policies = []
        for kk in range(1, k+1):
            policy, target_policy = get_level_k_policy(env, kk, M, i, u_range, opponent_conditional_policy, game_name=game_name)
            policies.append(policy)
            target_policies.append(target_policy)
        policy = GeneralizedMultiLevelPolicy(env.env_specs, policies=policies, agent_id=i, k=k, mu=mu)
        target_policy = GeneralizedMultiLevelPolicy(env.env_specs, policies=policies, agent_id=i, k=k, mu=mu, correct_tanh=correct_tanh)
    else:
        if k == 0:
            policy = DeterministicNNPolicy(env.env_specs,
                                           hidden_layer_sizes=(M, M),
                                           squash=squash, squash_func=squash_func, sampling=sampling,u_range=u_range, joint=False,
                                           agent_id=i)
            target_policy = DeterministicNNPolicy(env.env_specs,
                                                  hidden_layer_sizes=(M, M),
                                                  name='target_policy',
                                                  squash=squash, squash_func=squash_func, sampling=sampling,u_range=u_range, joint=False,
                                                  agent_id=i)
        if k > 0:
            policy, target_policy = get_level_k_policy(env, k, M, i, u_range, opponent_conditional_policy, game_name=game_name)




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
        conditional_policy=opponent_conditional_policy,
        plotter=plotter,
        policy_lr=1e-2,
        qf_lr=1e-2,
        joint=False,
        value_n_particles=16,
        kernel_fn=adaptive_isotropic_gaussian_kernel,
        kernel_n_particles=32,
        kernel_update_ratio=0.5,
        td_target_update_interval=1,
        discount=0.95,
        reward_scale=1,
        tau=0.01,
        save_full_state=False,
        k=k,
        aux=aux)
    return agent





def ddpg_agent(joint, opponent_modelling, model_name, i, env, M, u_range, base_kwargs, game_name='matrix'):
    # joint = True
    # opponent_modelling = False
    print(model_name)
    squash = True
    squash_func = tf.tanh
    sampling = False

    if 'particle' in game_name:
        squash_func = tf.nn.softmax
        sampling = True

    print(joint, opponent_modelling)
    pool = SimpleReplayBuffer(env.env_specs, max_replay_buffer_size=1e6, joint=joint, agent_id=i)
    policy = DeterministicNNPolicy(env.env_specs,
                                   hidden_layer_sizes=(M, M),
                                   squash=squash, squash_func=squash_func, sampling=sampling,u_range=u_range, joint=False,
                                   agent_id=i)
    target_policy = DeterministicNNPolicy(env.env_specs,
                                          hidden_layer_sizes=(M, M),
                                          name='target_policy',
                                          squash=squash, squash_func=squash_func,sampling=sampling, u_range=u_range, joint=False,
                                          agent_id=i)
    opponent_policy = None
    if opponent_modelling:
        opponent_policy = DeterministicNNPolicy(env.env_specs,
                                                hidden_layer_sizes=(M, M),
                                                name='opponent_policy',
                                                squash=squash, squash_func=squash_func, u_range=u_range, joint=True,
                                                opponent_policy=True,
                                                agent_id=i)
    maddpg = False
    # print(model_name)
    if 'MADDPG' in model_name:
        # print(MADDPG)
        maddpg = True
        model_name = 'MADDPG'
    qf = NNQFunction(env_spec=env.env_specs, hidden_layer_sizes=[M, M], joint=joint, agent_id=i, maddpg=maddpg)
    target_qf = NNQFunction(env_spec=env.env_specs, hidden_layer_sizes=[M, M], name='target_qf', joint=joint,
                            agent_id=i, maddpg=maddpg)
    plotter = None
    # print(model_name)
    agent = MADDPG(
        base_kwargs=base_kwargs,
        agent_id=i,
        name=model_name,
        env=env,
        pool=pool,
        qf=qf,
        target_qf=target_qf,
        policy=policy,
        target_policy=target_policy,
        opponent_policy=opponent_policy,
        plotter=plotter,
        policy_lr=1e-2,
        qf_lr=1e-2,
        joint=joint,
        opponent_modelling=opponent_modelling,
        td_target_update_interval=1,
        discount=0.95,
        reward_scale=1.,
        save_full_state=False)
    return agent





def rom_agent(model_name, i, env, M, u_range, base_kwargs, g=False, mu=1.5, game_name='matrix'):
    print(model_name)
    joint = False
    squash = True
    opponent_modelling = True
    squash_func = tf.tanh
    correct_tanh = True
    sampling = False
    # TODO deal with particle problem.
    if 'particle' in game_name:
        sampling = True
        squash = True
        squash_func = tf.nn.softmax
        correct_tanh = False

    pool = SimpleReplayBuffer(env.env_specs, max_replay_buffer_size=1e6, joint=joint, agent_id=i)

    opponent_policy = GaussianPolicy(env.env_specs,
                                      hidden_layer_sizes=(M, M),
                                      squash=True, joint=False,
                                      agent_id=i, name='opponent_policy')
    conditional_policy = GaussianConditionalPolicy(env.env_specs,
                                                   cond_policy=opponent_policy,
                                                   hidden_layer_sizes=(M, M),
                                                   name='gaussian_conditional_policy',
                                                   opponent_policy=False,
                                                   squash=True, joint=False,
                                                   agent_id=i)
    with tf.variable_scope('target_levelk_{}'.format(i), reuse=True):
        target_opponent_policy = GaussianPolicy(env.env_specs,
                                         hidden_layer_sizes=(M, M),
                                         squash=True, joint=False,
                                         agent_id=i, name='target_opponent_policy')
        target_conditional_policy = GaussianConditionalPolicy(env.env_specs,
                                                       cond_policy=target_opponent_policy,
                                                       hidden_layer_sizes=(M, M),
                                                       name='target_gaussian_conditional_policy',
                                                       opponent_policy=False,
                                                       squash=True, joint=False,
                                                       agent_id=i)


    joint_qf = NNJointQFunction(env_spec=env.env_specs, hidden_layer_sizes=[M, M], joint=joint, agent_id=i)
    target_joint_qf = NNJointQFunction(env_spec=env.env_specs, hidden_layer_sizes=[M, M], name='target_joint_qf', agent_id=i)

    plotter = None

    agent = ROMMEO(
        base_kwargs=base_kwargs,
        agent_id=i,
        env=env,
        pool=pool,
        joint_qf=joint_qf,
        target_joint_qf=target_joint_qf,
        policy=conditional_policy,
        opponent_policy=opponent_policy,
        target_policy=target_conditional_policy,
        plotter=plotter,
        policy_lr=1e-2,
        qf_lr=1e-2,
        joint=True,
        value_n_particles=16,
        kernel_fn=adaptive_isotropic_gaussian_kernel,
        kernel_n_particles=32,
        kernel_update_ratio=0.5,
        td_target_update_interval=1,
        discount=0.95,
        reward_scale=1,
        tau=0.01,
        save_full_state=False,
        opponent_modelling=True)
    return agent