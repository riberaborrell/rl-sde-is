import numpy as np
import torch

def get_epsilon_greedy_discrete_action(env, model, state, epsilon):

    # pick greedy action (exploitation)
    if np.random.rand() > epsilon:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32)
            q_values = model.forward(state_tensor).numpy()
            idx_action = np.argmax(q_values)

    # pick random action (exploration)
    else:
        idx_action = np.random.randint(env.n_actions)

    action = env.action_space_h[[idx_action]]

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
    actions = env.get_greedy_actions(q_table)

    return q_table, v_table, a_table, actions

def compute_tables_actor_critic(env, actor, critic):

    # discretized states
    state_space_h = torch.FloatTensor(env.state_space_h).unsqueeze(dim=1)

    with torch.no_grad():
        actions = actor.forward(state_space_h)
        v_table = critic.forward(torch.hstack((state_space_h, actions)))

    return v_table, actions


def test_model(env, model, batch_size=10):

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
            state, r, done = env.step(state, action)

            # save reward + time steps
            ep_ret += r
            ep_len += 1

        ep_rets.append(ep_ret)
        ep_lens.append(ep_len)

    return np.mean(ep_rets), np.mean(ep_lens)
