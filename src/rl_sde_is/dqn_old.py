
def sample_trajectories_buffer_vectorized(env, model, replay_buffer, batch_size, n_max, epsilon):

    # are episodes done
    already_done = np.full((batch_size,), False)
    done = np.full((batch_size,), False)

    # initialize episodes
    states = np.full((batch_size, env.d), env.state_init)

    # sample episodes
    for n in np.arange(n_max):

        # get action being epsilon greedy 
        actions_idx = get_epsilon_greedy_discrete_action_vectorized(env, model, states, epsilon)
        actions = np.expand_dims(env.action_space_h[actions_idx], axis=1)

        # step dynamics forward
        next_states, rewards, done, _ = env.step(states, actions, reward_type='state-action-next-state')

        # store tuple
        idx = np.where(np.invert(already_done))[0]
        replay_buffer.store_vectorized(states[idx], actions_idx[idx], rewards[idx],
                                       next_states[idx], done[idx])

        # get indices of episodes which are new to the target set
        _ = env.get_new_in_ts_idx(done, already_done)

        # stop if all episodes already in target set
        if already_done.all() == True:
           break

        # update states
        states = next_states

    # preallocate lists to hold results
    batch_states = np.zeros([batch_size, env.state_space_dim], dtype=np.float32)
    batch_next_states = np.zeros([batch_size, env.state_space_dim], dtype=np.float32)
    batch_actions = np.zeros(batch_size, dtype=np.int64)
    batch_rewards = np.zeros(batch_size, dtype=np.float32)
    batch_done = np.zeros(batch_size, dtype=bool)

    batch_counter = 0
    batch_discounted_returns = []
    batch_idx = 0

    total_returns = []
    total_time_steps = []

    # set epsilon
    epsilon = eps_init

    for ep in np.arange(n_episodes):

        # reset state
        state = env.reset()

        # preallocate rewards for the episode
        ep_rewards = []

        # time step
        k = 0

        print(k)

        done = False
        while done == False:

            # save state
            batch_states[batch_idx] = state.copy()

            # get action following q-values
            #action = get_epsilon_greedy_action(env, model, state, epsilon)
            _, action = get_epsilon_greedy_discrete_action(env, model, state, epsilon)

            # next step
            new_state, r, done, _ = env.step(state, action)
            k += 1

            # save action and reward
            batch_actions[batch_idx] = action
            batch_rewards[batch_idx] = r
            batch_done[batch_idx] = done

            # update states
            state = new_state

            # update epsilon
            #epsilon = 1 + (eps_final - 1)*min(1, t/finish_decay)
            epsilon = 0.

        # update batch data
        batch_discounted_returns.extend(discount_cumsum(ep_rewards, gamma))
        batch_counter += 1
        total_returns.append(sum(ep_rewards))
        total_time_steps.append(k)

        # batch is complete
        if batch_counter == batch_size:

            # update parameters 
            batch = (batch_states, batch_next_states, batch_actions, batch_rewards, batch_done)
            step_loss = update_parameters(optimizer, model, target_model, batch, gamma)

            # update network
            target_model.load_state_dict(model.state_dict())

            # reset batch
            batch_states = []
            batch_actions = []
            batch_discounted_returns = []
            batch_counter = 0

            # print running average
            run_avg_msg = 'ep: {}, run avg returns: {:.2f}, run avg time steps: {:.2f}'.format(
                ep + 1,
                np.mean(total_returns[-batch_size:]),
                np.mean(total_time_steps[-batch_size:]),
            )
            print(run_avg_msg)

    return model
