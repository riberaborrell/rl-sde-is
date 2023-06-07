import numpy as np
import torch

def compute_running_mean(array, run_window=10):
    ''' computes the running mean / moving average of the given array along the given running window.
    '''
    return np.array([
        np.mean(array[i-run_window:i+1]) if i > run_window
        else np.mean(array[:i+1]) for i in range(len(array))
    ])

def compute_running_variance(array, run_window=10):
    ''' computes the running variance of the given array along the given running window.
    '''
    return np.array([
        np.var(array[i-run_window:i+1]) if i > run_window
        else np.var(array[:i+1]) for i in range(len(array))
    ])

def get_epsilon_greedy_discrete_action(env, model, state, epsilon):

    # pick greedy action (exploitation)
    if np.random.rand() > epsilon:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32)
            q_values = model.forward(state_tensor).numpy()
            idx_action = np.argmax(q_values, axis=1)
            action = env.action_space_h[[idx_action]]

    # pick random action (exploration)
    else:
        idx_action = np.random.randint(env.n_actions)
        action = env.action_space_h[[[idx_action]]]

    return idx_action, action

def get_epsilon_greedy_continuous_action(env, model, state, epsilon):

    # pick greedy action (exploitation)
    if np.random.rand() > epsilon:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32)
            return model.forward(state_tensor).numpy()

    # pick random action (exploration)
    else:
        return np.random.uniform(env.action_space_low, env.action_space_high, (1,))

def compute_v_table(env, model):
    states = torch.FloatTensor(env.state_space_h).unsqueeze(dim=1)

    # compute v table
    with torch.no_grad():
        v_table = model.forward(states).numpy()

    return v_table

def compute_tables_discrete_actions(env, model):

    states = torch.FloatTensor(env.state_space_h).unsqueeze(dim=1)

    # compute q table
    with torch.no_grad():
        q_table = model.forward(states).numpy()

    # compute value function
    v_table = np.max(q_table, axis=1)

    # compute advantage table
    a_table = q_table - np.expand_dims(v_table, axis=1)

    # compute greedy actions
    greedy_actions = env.get_greedy_actions(q_table)

    return q_table, v_table, a_table, greedy_actions

def compute_tables_continuous_actions(env, model):

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

    # compute value function
    v_table = np.max(q_table, axis=1)

    # compute advantage table
    a_table = q_table - np.expand_dims(v_table, axis=1)

    # compute greedy actions
    idx_actions = np.argmax(q_table, axis=1)
    greedy_actions = env.action_space_h[idx_actions]

    return q_table, v_table, a_table, greedy_actions

def compute_det_policy_actions(env, model, states):
    with torch.no_grad():
        return model.forward(states).numpy()

def compute_tables_critic(env, critic):

    # discretized states and actions
    state_space_h = torch.FloatTensor(env.state_space_h)
    action_space_h = torch.FloatTensor(env.action_space_h)
    grid_states, grid_actions = torch.meshgrid(state_space_h, action_space_h, indexing='ij')

    #inputs = torch.empty((env.n_states, env.n_actions, 2))
    #inputs[:, :, 0] = grid_states
    #inputs[:, :, 1] = grid_actions
    #inputs = inputs.reshape(env.n_states * env.n_actions, 2)
    grid_states = grid_states.reshape(env.n_states * env.n_actions, 1)
    grid_actions = grid_actions.reshape(env.n_states * env.n_actions, 1)

    # compute q table
    with torch.no_grad():
        q_table = critic.forward(grid_states, grid_actions).numpy().reshape(env.n_states, env.n_actions)
        #q_table = critic.forward(grid_states, grid_actions).numpy()

    # compute value function
    v_table = np.max(q_table, axis=1)

    # compute advantage table
    a_table = q_table - np.expand_dims(v_table, axis=1)

    # compute greedy actions
    idx_actions = np.argmax(q_table, axis=1)
    greedy_actions = env.action_space_h[idx_actions]

    return q_table, v_table, a_table, greedy_actions


def compute_tables_actor_critic(env, actor, critic):

    # discretized states
    state_space_h = torch.FloatTensor(env.state_space_h).unsqueeze(dim=1)

    with torch.no_grad():
        actions = actor.forward(state_space_h)
        v_table = critic.forward(state_space_h, actions)
        #v_table = critic.forward(torch.hstack((state_space_h, actions)))

    return v_table, actions

def compute_v_value_critic(env, critic, state):
    action_space_h = torch.FloatTensor(env.action_space_h).unsqueeze(dim=1)
    states = torch.ones_like(action_space_h) * torch.FloatTensor(state)
    inputs = torch.hstack((states, action_space_h))
    with torch.no_grad():
        q_values = critic.forward(inputs).numpy()
    return np.max(q_values)


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

def test_policy_vectorized(env, model, batch_size=10, k_max=10**5, policy_opt=None):

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
        idx_states = env.get_state_idx(states)
        actions_opt = policy_opt[idx_states]

        # computer running u l2 error
        if policy_opt is not None:
            ep_policy_l2_error_t += (np.linalg.norm(actions - actions_opt, axis=1) ** 2) * env.dt

        # get indices of episodes which are new to the target set
        idx = env.get_idx_new_in_ts(done, already_done)

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

    if policy_opt is not None:
        return np.mean(ep_rets), np.var(ep_rets), np.mean(ep_lens), np.mean(ep_policy_l2_error_fht)
    else:
        return np.mean(ep_rets), np.var(ep_rets), np.mean(ep_lens)
