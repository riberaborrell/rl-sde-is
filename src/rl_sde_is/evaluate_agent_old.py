def evaluate_policy():
    # preallocate returns
    returns = []
    time_steps = []

    # preallocate trajectory
    ep_states = np.empty((0, env.d))
    ep_actions = np.empty((0, env.d))
    ep_rewards = np.empty((0,))
    # sample trajectories
    for ep in np.arange(n_episodes):

        # reset environment
        state = env.reset()

        # terminal state flag
        done = False

        # initialize episodic return
        ep_return = 0.

        for k in range(n_steps_lim):

            # interrupt if we are in a terminal state
            if done:
                break

            # copy state
            state_copy = state.copy()

            # take a random action
            if agent == 'random':
                action = np.random.rand(1, env.d) * env.action_space_bounds[1]
            elif agent == 'not-controlled':
                action = np.zeros((1, env.d))
            elif agent == 'hjb':
                idx = env.get_state_idx(state)
                action = np.expand_dims(sol_hjb.u_opt[idx], axis=0)

            # step dynamics forward
            next_state, r, done, _ = env.step(state, action)

            #print('step: {}, state: {:.1f}, action: {:.1f}, reward: {:.3f}'
            #      ''.format(k, state_copy[0], action[0], r))

            # compute return
            ep_return += r

            # save first trajectory
            if save_traj and ep == 0:
                ep_states = np.vstack((ep_states, state))
                ep_actions = np.vstack((ep_actions, action))
                ep_rewards = np.append(ep_rewards, r)

            # update state
            state = next_state

        # save episode info
        returns.append(ep_return)
        time_steps.append(k)

