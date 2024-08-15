import copy

import numpy as np
import torch
import torch.optim as optim

from gym_sde_is.wrappers.tabular_env import TabularEnv

from rl_sde_is.utils.tabular_methods import compute_value_advantage_and_greedy_policy, \
                                            compute_value_advantage_and_greedy_actions
from rl_sde_is.utils.path import save_data
from rl_sde_is.utils.plots import *

def evaluate_det_policy_model(env, model):
    state_space_h = torch.FloatTensor(env.state_space_h)
    with torch.no_grad():
        return model.forward(state_space_h).numpy().reshape(env.n_states, env.d)

def evaluate_stoch_policy_model(env, model):
    state_space_h = torch.FloatTensor(env.state_space_h)
    with torch.no_grad():
        means, stds = model.mean_and_std(state_space_h)
        return means.numpy(), stds.numpy()

def evaluate_value_function_model(env, model):
    state_space_h = torch.FloatTensor(env.state_space_h)
    with torch.no_grad():
        return model.forward(state_space_h).numpy().reshape(env.n_states)

def evaluate_qvalue_function_model_1d(env, model):
    # discretized states and actions
    state_space_h = torch.FloatTensor(env.state_space_h.squeeze())
    action_space_h = torch.FloatTensor(env.action_space_h.squeeze())
    grid_states, grid_actions = torch.meshgrid(state_space_h, action_space_h, indexing='ij')

    n_states = env.state_space_h.shape[0]
    n_actions = env.action_space_h.shape[0]

    grid_states = grid_states.reshape(n_states * n_actions, 1)
    grid_actions = grid_actions.reshape(n_states * n_actions, 1)

    # compute q table
    with torch.no_grad():
        return model.forward(grid_states, grid_actions).numpy().reshape(n_states, n_actions)

def get_epsilon_greedy_discrete_action(env, model, state, epsilon):

    # pick greedy action (exploitation)
    if np.random.rand() > epsilon:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32)
            q_values = model.forward(state_tensor).numpy()
            action_idx = np.argmax(q_values, axis=1)
            action = env.action_space_h[[action_idx]]

    # pick random action (exploration)
    else:
        action_idx = np.random.randint(env.n_actions)
        action = env.action_space_h[[[action_idx]]]

    return action_idx, action

def get_epsilon_greedy_discrete_action_vectorized(env, model, states, epsilon):

    batch_size = states.shape[0]
    actions_idx = np.empty(batch_size, dtype=np.int32)

    #if np.random.rand() > epsilon:
    bools = np.random.rand(batch_size) > epsilon
    idx = np.where(bools)[0]
    idx_c = np.where(np.invert(bools))[0]

    # exploitation
    with torch.no_grad():
        states_tensor = torch.tensor(states[idx], dtype=torch.float32)
        q_values = model.forward(states_tensor).numpy()
        actions_idx[idx] = np.argmax(q_values, axis=1)

    # exploration
    actions_idx[idx_c] = np.random.randint(0, env.n_actions, idx_c.shape[0])

    return actions_idx

def get_epsilon_greedy_continuous_action(env, model, state, epsilon):

    # pick greedy action (exploitation)
    if np.random.rand() > epsilon:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32)
            return model.forward(state_tensor).numpy()

    # pick random action (exploration)
    else:
        return np.random.uniform(env.action_space_bounds[0], env.action_space_bounds[1], (1,))



def compute_q_table_continuous_actions_1d(env, model):

    # discretized states and actions
    state_space_h = torch.FloatTensor(env.state_space_h)
    action_space_h = torch.FloatTensor(env.action_space_h)
    grid_states, grid_actions = torch.meshgrid(state_space_h, action_space_h, indexing='ij')

    inputs = torch.empty((env.n_states, env.n_actions, 2))
    inputs[:, :, 0] = grid_states
    inputs[:, :, 1] = grid_actions
    inputs = inputs.reshape(env.n_states * env.n_actions, 2)

    # compute q table
    with torch.no_grad():
        q_table = model.forward(inputs[:, 0:1], inputs[:, 1:2]).numpy().reshape(env.n_states, env.n_actions)

    return q_table

def compute_tables_discrete_actions_1d(env, model):

    states = torch.FloatTensor(env.state_space_h).unsqueeze(dim=1)

    # compute q table
    with torch.no_grad():
        q_table = model.forward(states).numpy()

    # compute value table, advantage table and greedy actions
    v_table, a_table, actions_idx = compute_value_advantage_and_greedy_actions(q_table)

    # compute greedy actions
    greedy_actions = env.get_greedy_actions(q_table)

    return q_table, v_table, a_table, greedy_actions

def compute_tables_continuous_actions_1d(env, model):

    # discretized states and actions
    state_space_h = torch.FloatTensor(env.state_space_h)
    action_space_h = torch.FloatTensor(env.action_space_h)
    grid_states, grid_actions = torch.meshgrid(state_space_h, action_space_h, indexing='ij')

    inputs = torch.empty((env.n_states, env.n_actions, 2))
    inputs[:, :, 0] = grid_states
    inputs[:, :, 1] = grid_actions
    inputs = inputs.reshape(env.n_states * env.n_actions, 2)

    # compute q table
    with torch.no_grad():
        q_table = model.forward(inputs).numpy().reshape(env.n_states, env.n_actions)

    # compute value table, advantage table and greedy actions
    v_table, a_table, actions_idx = compute_value_advantage_and_greedy_actions(q_table)

    # compute greedy actions
    greedy_actions = env.action_space_h[actions_idx]

    return q_table, v_table, a_table, greedy_actions

def compute_det_policy_actions(env, model, states):
    with torch.no_grad():
        return model.forward(states).numpy()


def compute_tables_critic_1d(env, critic):

    q_table = evaluate_qvalue_function_model_1d(env, critic)

    # compute value table, advantage table and greedy actions
    v_table, a_table, actions_idx = compute_value_advantage_and_greedy_actions(q_table)

    # compute greedy actions
    greedy_actions = env.action_space_h[actions_idx]

    return q_table, v_table, a_table, greedy_actions

def compute_tables_critic_2d(env, critic):

    env.discretize_state_action_space(env.h_state, env.h_action)

    grid_states = torch.tensor(env.state_action_space_h[:, :2], dtype=torch.float32)
    grid_actions = torch.tensor(env.state_action_space_h[:, 2:], dtype=torch.float32)

    # compute q table
    with torch.no_grad():
        q_table = critic.forward(grid_states, grid_actions).numpy().reshape(env.n_states, env.n_actions)

    # compute value table, advantage table and greedy action indices
    v_table, a_table, actions_idx = compute_value_advantage_and_greedy_actions(q_table)

    # greedy actions
    greedy_actions = env.action_space_h.reshape(env.n_actions, env.d)[actions_idx]

    # reshape
    q_table = q_table.reshape(env.n_states_i1, env.n_states_i2, env.n_actions_i1, env.n_actions_i2)
    v_table = v_table.reshape(env.n_states_i1, env.n_states_i2)
    a_table = a_table.reshape(env.n_states_i1, env.n_states_i2, env.n_actions_i1, env.n_actions_i2)
    greedy_actions = greedy_actions.reshape(env.n_states_i1, env.n_states_i2, env.d)

    return q_table, v_table, a_table, greedy_actions


def compute_tables_actor_critic_1d(env, actor, critic):

    # discretized states
    state_space_h = torch.FloatTensor(env.state_space_h).unsqueeze(dim=1)

    with torch.no_grad():
        actions = actor.forward(state_space_h)
        v_table = critic.forward(state_space_h, actions).numpy()

    actor_policy = actions.numpy().squeeze()

    return v_table, actor_policy

def compute_v_value_critic_1d(env, critic, state):
    action_space_h = torch.FloatTensor(env.action_space_h).unsqueeze(dim=1)
    states = torch.ones_like(action_space_h) * torch.FloatTensor(state)
    inputs = torch.hstack((states, action_space_h))
    with torch.no_grad():
        q_values = critic.forward(inputs).numpy()
    return np.max(q_values)

def load_dp_tables_data(env, dt=1e-4, h_state=1e-2, h_action=1e-2):
    from rl_sde_is.dynamic_programming.compute_dp_tables import dynamic_programming_tables

    # initialize environment
    env_dp = copy.deepcopy(env)

    # discretize state and action space
    env_dp = TabularEnv(env_dp, h_state, h_action)
    env_dp.discretize_state_action_space(h_state, h_action)

    # run dynamic programming q-value iteration
    data = dynamic_programming_tables(env_dp, load=True)

    return env_dp, data

def train_deterministic_policy_from_hjb(env, policy, policy_opt, load=False):

    from rl_sde_is.utils.plots import initialize_det_policy_1d_figure

    # optimizer
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    # train
    n_iterations = int(1e4)

    # minibatch size
    batch_size = int(1e3)
    for i in range(n_iterations):

        idx = np.random.randint(0, env.n_states, batch_size)

        # sample data
        states = torch.tensor(env.state_space_h[idx], dtype=torch.float32)

        # compute q values
        means = policy(states)

        # targets
        mean_targets = torch.tensor(policy_opt[idx], dtype=torch.float32)

        # compute mse loss
        loss = (means - mean_targets).pow(2).mean()

        # compute gradient and update params
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % int(1e3) == 0:
            print('iteration: {:d}, loss: {:1.4e}'.format(i, loss))

    print('Policy mean trained to be the optimal policy')
    return policy

def train_stochastic_policy_from_hjb(env, policy, policy_opt, load=False):

    from rl_sde_is.utils.plots import initialize_det_policy_1d_figure

    # optimizer
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    # train
    n_iterations = int(1e4)

    # minibatch size
    batch_size = int(1e3)
    for i in range(n_iterations):

        idx = np.random.randint(0, env.n_states, batch_size)

        # sample data
        states = torch.tensor(
            env.state_space_h.reshape(env.n_states, env.d)[idx], dtype=torch.float32,
        )

        # compute q values
        means = policy.mean(states)

        # targets
        mean_targets = torch.tensor(
            policy_opt.reshape(env.n_states, env.d)[idx], dtype=torch.float32,
        )

        # compute mse loss
        loss = (means - mean_targets).pow(2).mean()

        # compute gradient and update params
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % int(1e3) == 0:
            print('iteration: {:d}, loss: {:1.4e}'.format(i, loss))

    print('Policy mean trained to be the optimal policy')
    return policy

def train_critic_discrete_from_dp(env, critic, value_function_opt, policy_opt, load=False):
    from rl_sde_is.utils.plots import initialize_qvalue_function_1d_figure, \
                                initialize_advantage_function_1d_figure, \
                                update_imshow_figure

    # load q value table from dynamic programming
    env_dp, data = load_dp_tables_data(env, dt=1e-3, h_state=1e-2, h_action=1e-2)

    # load critic if already trained
    if load and 'q_function_approx' in data:
        return data['q_function_approx']
        print('Critic load to be the actual q-value function (from dp)')

    q_table_dp = data['q_table']
    _, _, policy_dp = compute_value_advantage_and_greedy_policy(env_dp, q_table_dp)

    # initialize figures
    q_table, _, a_table, policy = compute_tables_critic_1d(env, critic)
    im_q = initialize_qvalue_function_1d_figure(env, q_table)
    im_a, line = initialize_advantage_function_1d_figure(env, a_table, policy_opt, policy)
    #im_a, line = initialize_advantage_function_1d_figure(env, a_table, policy_dp, policy)

    # optimizer
    optimizer = optim.Adam(critic.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    # train q-value function parameters
    n_iterations = int(1e5)
    batch_size = int(1e3)

    for i in range(n_iterations):

        # sample data
        state_actions_idx = np.random.randint(0, env_dp.n_states_actions, batch_size)
        state_actions = env_dp.state_action_space_h_flat[state_actions_idx]
        states = torch.tensor(state_actions[:, 0], dtype=torch.float32).unsqueeze(dim=1)
        actions = torch.tensor(state_actions[:, 1], dtype=torch.float32).unsqueeze(dim=1)

        # targets
        q_values_target = torch.tensor(
            q_table_dp.reshape(env_dp.n_states_actions), dtype=torch.float32
        )[state_actions_idx]

        # compute q values
        q_values = critic.forward(states, actions)

        # compute mse loss
        loss = ((q_values - q_values_target)**2).mean()

        # compute gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if i % int(1e3) == 0:
            print('iteration: {:d}, loss: {:1.4e}, lr: {:1.4e}'.format(i, loss, get_lr(optimizer)))
            q_table, _, a_table, policy = compute_tables_critic_1d(env, critic)
            update_imshow_figure(env, q_table, im_q)
            update_advantage_function_1d_figure(env, a_table, im_a, policy, line)

    # save
    data['q_function_approx'] = critic
    save_data(data, data['dir_path'])

    print('Critic trained to be the actual q-value function (from dp)')
    return critic

def train_critic_from_dp(env, critic, value_function_opt, policy_opt, load=False):

    from rl_sde_is.utils.plots import initialize_qvalue_function_1d_figure, \
                                initialize_advantage_function_1d_figure, \
                                update_imshow_figure

    # load q value table from dynamic programming
    env_dp, data = load_dp_tables_data(env, dt=1e-3, h_state=1e-2, h_action=1e-2)

    # load critic if already trained
    if load and 'q_function_approx' in data:
        return data['q_function_approx']
        print('Critic load to be the actual q-value function (from dp)')

    q_table_dp = data['q_table']
    breakpoint()
    _, _, policy_dp = compute_value_advantage_and_greedy_policy(env_dp, q_table_dp)

    # initialize figures
    q_table, _, a_table, policy = compute_tables_critic_1d(env, critic)
    im_q = initialize_qvalue_function_1d_figure(env, q_table)
    im_a, line = initialize_advantage_function_1d_figure(env, a_table, policy_opt, policy)
    #im_a, line = initialize_advantage_function_1d_figure(env, a_table, policy_dp, policy)

    # optimizer
    optimizer = optim.Adam(critic.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    # train q-value function parameters
    n_iterations = int(1e3)
    batch_size = int(1e3)

    for i in range(n_iterations):

        # sample data
        state_actions_idx = np.random.randint(0, env_dp.n_states_actions, batch_size)
        state_actions = env_dp.state_action_space_h_flat[state_actions_idx]
        states = torch.tensor(state_actions[:, 0], dtype=torch.float32).unsqueeze(dim=1)
        actions = torch.tensor(state_actions[:, 1], dtype=torch.float32).unsqueeze(dim=1)

        # targets
        q_values_target = torch.tensor(
            q_table_dp.reshape(env_dp.n_states_actions), dtype=torch.float32
        )[state_actions_idx]

        # compute q values
        q_values = critic.forward(states, actions)

        # compute mse loss
        loss = ((q_values - q_values_target)**2).mean()

        # compute gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if i % int(1e3) == 0:
            print('iteration: {:d}, loss: {:1.4e}, lr: {:1.4e}'.format(i, loss, get_lr(optimizer)))
            q_table, _, a_table, policy = compute_tables_critic_1d(env, critic)
            update_imshow_figure(env, q_table, im_q)
            update_advantage_function_1d_figure(env, a_table, im_a, policy, line)

    # save
    data['q_function_approx'] = critic
    save_data(data, data['dir_path'])

    print('Critic trained to be the actual q-value function (from dp)')
    return critic

def train_dueling_critic_from_dp(env, critic_v, critic_a, value_function_opt, policy_opt, load=False):

    from rl_sde_is.utils.plots import initialize_value_function_1d_figure, \
                                initialize_qvalue_function_1d_figure, \
                                initialize_advantage_function_1d_figure, \
                                update_value_function_1d_figure, \
                                update_imshow_figure

    # load q value table from dynamic programming
    env_dp, data = load_dp_tables_data(env, dt=1e-4, h_state=1e-2, h_action=1e-2)

    # load critics if already trained
    if load and 'v_function_approx' in data and 'a_function_approx' in data:
        return data['v_function_approx'], data['a_function_approx']
        print('Critic load to be the actual q-value function (from dp)')

    # compute v and a tables
    q_table_dp = data['q_table']
    #v_table_dp = np.max(q_table_dp, axis=1)
    #a_table_dp = q_table_dp - np.expand_dims(v_table_dp, axis=1)
    v_table_dp, a_table_dp, policy_dp = compute_value_advantage_and_greedy_policy(env_dp, q_table_dp)

    # initialize figures
    v_table = compute_v_table_1d(env, critic_v)
    l_v = initialize_value_function_1d_figure(env, v_table, value_function_opt)

    # optimizer
    optimizer_v = optim.Adam(critic_v.parameters(), lr=1e-3)

    # trainning data
    batch_size = int(1e2)
    states_idx = np.random.randint(0, env_dp.n_states, batch_size)
    states = torch.tensor(env_dp.state_space_h[states_idx], dtype=torch.float32).unsqueeze(dim=1)

    # targets
    v_values_target = torch.tensor(v_table_dp[states_idx], dtype=torch.float32)

    # train value function parameters
    n_iterations = int(2e3)

    for i in range(n_iterations):

        # compute q values
        v_values = critic_v.forward(states).squeeze()

        # compute mse loss
        loss_v = ((v_values - v_values_target)**2).mean()

        # compute gradient
        optimizer_v.zero_grad()
        loss_v.backward()
        optimizer_v.step()

        if i % int(1e2) == 0:
            print('iteration: {:d}, loss: {:1.4e}'.format(i, loss_v))
            v_table = compute_v_table_1d(env, critic_v)
            update_value_function_1d_figure(env, v_table, l_v)

    print('Critic for the value function trained to be the actual q-value function (from dp)')

    # initialize figures
    a_table = compute_q_table_continuous_actions_1d(env, critic_a)
    #q_table, _, a_table, policy = compute_tables_critic_1d(env, critic)
    q_table = a_table + v_table
    im_q = initialize_qvalue_function_1d_figure(env, q_table)
    im_a = initialize_advantage_function_1d_figure(env, a_table, policy_opt)

    # optimizer
    optimizer_a = optim.Adam(critic_a.parameters(), lr=1e-3)

    # train value function parameters
    batch_size = int(1e4)
    n_iterations = int(5e3)

    for i in range(n_iterations):

        # sample data
        state_actions_idx = np.random.randint(0, env_dp.n_states_actions, batch_size)
        state_actions = env_dp.state_action_space_h_flat[state_actions_idx]
        states = torch.tensor(state_actions[:, 0], dtype=torch.float32).unsqueeze(dim=1)
        actions = torch.tensor(state_actions[:, 1], dtype=torch.float32).unsqueeze(dim=1)

        # targets
        a_values_target = torch.tensor(
            a_table_dp.reshape(env_dp.n_states_actions), dtype=torch.float32
        )[state_actions_idx]

        # compute q values
        a_values = critic_a.forward(states, actions)

        # compute mse loss
        loss_a = ((a_values - a_values_target)**2).mean()

        # compute gradient
        optimizer_a.zero_grad()
        loss_a.backward()
        optimizer_a.step()

        if i % int(1e3) == 0:
            print('iteration: {:d}, loss: {:1.4e}'.format(i, loss_a))
            v_table = compute_v_table_1d(env, critic_v)
            a_table = compute_q_table_continuous_actions_1d(env, critic_a)
            q_table = a_table + v_table
            update_imshow_figure(env, q_table, im_q)
            update_imshow_figure(env, a_table, im_a)

    print('Critic trained to be the actual q-value function (from dp)')

    # save
    data['v_function_approx'] = critic_v
    data['a_function_approx'] = critic_a
    save_data(data, data['dir_path'])

    return critic_v, critic_a


def sample_trajectories_buffer(env, policy, replay_buffer, n_episodes, n_steps_lim):

    for k in np.arange(n_episodes):

        # initialization
        state, _ = env.reset()

        # terminal state flag
        done = False

        # sample trajectory
        for n in np.arange(n_steps_lim):

            # interrupt if we are in a terminal state
            if done:
                break

            # get action following the actor
            with torch.no_grad():
                action = policy.forward(torch.FloatTensor(state)).numpy()

            # env step
            next_state, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # store tuple
            replay_buffer.store(state, action, r, next_state, done)

            # update state
            state = next_state


def sample_trajectories_buffer_vect(env, policy, replay_buffer, n_episodes, n_steps_lim):

    # reset environment
    states, _ = env.reset(batch_size=n_episodes)#, seed=seed)

    done = np.full((n_episodes,), False)
    while not done.all():

        # take the action following the policy
        states_torch = torch.tensor(states, dtype=torch.float32)
        with torch.no_grad():
            actions = policy(states_torch).numpy()

        # step dynamics forward
        next_states, rewards, _, truncated, _ = env.step_vect(actions)
        done = np.logical_or(env.been_terminated, truncated)

        # store experiences at given time step
        idx = np.where(~env.been_terminated | env.new_terminated)[0]
        replay_buffer.store_vectorized(states[idx], actions[idx], rewards[idx],
                                       next_states[idx], done[idx])

        # update states
        states = next_states


def test_policy(env, model, batch_size=10):

    # preallocate returns and time steps
    ep_rets, ep_lens = [], []

    # sample trajectories
    for _ in range(batch_size):

        # reset
        state, r, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        while not(done):

            # get action following the deterministic policy model
            with torch.no_grad():
                action = model.forward(torch.FloatTensor(state)).numpy()

            # step forwards dynamics
            state, r, done, _ = env.step(state, action)

            # save reward + time steps
            ep_ret += r
            ep_len += 1

        ep_rets.append(ep_ret)
        ep_lens.append(ep_len)

    return np.mean(ep_rets), np.var(ep_rets), np.mean(ep_lens)

def test_policy_vectorized(env, model, batch_size=10, k_max=10**7, policy_opt=None):

    # preallocate returns and time steps
    total_rewards = np.zeros(batch_size)
    ep_rets = np.empty(batch_size)
    ep_lens = np.empty(batch_size)

    # preallocate u l2 error array
    if policy_opt is not None:
        ep_policy_l2_error_fht = np.empty(batch_size)
        ep_policy_l2_error_t = np.zeros(batch_size)

    # are episodes done
    already_done = np.full((batch_size,), False)
    done = np.full((batch_size,), False)

    # initialize episodes
    states = np.full((batch_size, env.d), env.state_init)

    # sample episodes
    for k in np.arange(k_max):

        # actions
        with torch.no_grad():
            actions = model.forward(torch.FloatTensor(states)).numpy()

        # step dynamics forward
        next_states, rewards, done, dbt = env.step(states, actions)

        # update total rewards for all trajectories
        total_rewards += np.squeeze(rewards)

        # hjb control
        states_idx = env.get_state_idx(states)
        actions_opt = policy_opt[states_idx]

        # computer running u l2 error
        if policy_opt is not None:
            ep_policy_l2_error_t += (np.linalg.norm(actions - actions_opt, axis=1) ** 2) * env.dt

        # get indices of episodes which are new to the target set
        idx = env.get_new_in_ts_idx(done, already_done)

        # if there are episodes which are done
        if idx.shape[0] != 0:

            # fix episode returns
            ep_rets[idx] = total_rewards[idx]

            # fix episode time steps
            ep_lens[idx] = k

            # fix l2 error
            if policy_opt is not None:
                ep_policy_l2_error_fht[idx] = ep_policy_l2_error_t[idx]

        # stop if xt_traj in target set
        if already_done.all() == True:
           break

        # update states
        states = next_states

    if not already_done.all() and policy_opt is not None:
        return np.nan, np.nan, np.nan, np.nan
    elif not already_done.all():
        return np.nan, np.nan, np.nan

    if policy_opt is not None:
        return np.mean(ep_rets), np.var(ep_rets), np.mean(ep_lens), np.mean(ep_policy_l2_error_fht)
    else:
        return np.mean(ep_rets), np.var(ep_rets), np.mean(ep_lens)

def estimate_fht_vectorized(env, model, batch_size=int(1e5), k_max=10**7):

    # preallocate time steps
    ep_fhts = np.empty(batch_size, dtype=np.int32)

    # are episodes done
    already_done = np.full((batch_size,), False)
    done = np.full((batch_size,), False)

    # initialize episodes
    states = np.full((batch_size, env.d), env.state_init)

    # freeze model to save computational effort 
    for param in model.parameters():
        param.requires_grad = False

    # sample episodes
    for k in np.arange(k_max):

        # actions
        actions = model.forward(torch.FloatTensor(states)).numpy()

        # step dynamics forward
        next_states, _, done, _ = env.step(states, actions)

        # get indices of episodes which are new to the target set
        idx = env.get_new_in_ts_idx(done, already_done)

        # if there are episodes which are done
        if idx.shape[0] != 0:

            # fix episode time steps
            ep_fhts[idx] = k

        # stop if xt_traj in target set
        if already_done.all() == True:
           break

        # update states
        states = next_states

    # unfreeze model
    for param in model.parameters():
        param.requires_grad = True

    return np.mean(env.dt * ep_fhts)
