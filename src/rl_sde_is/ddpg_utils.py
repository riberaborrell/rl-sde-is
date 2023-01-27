
def pre_train_critic2(env, critic):

    optimizer = optim.Adam(critic.parameters(), lr=1e-6)

    # batch size
    batch_size = 10**3

    # number of iterations
    n_iterations = 10**3

    env.state_space_low = -2
    env.state_space_high = 2
    env.action_space_low = -2
    env.action_space_high = 2

    states = torch
    def sample_data_points():

        # data points in the target set
        states_ts = (env.lb - env.rb) * torch.rand(batch_size_ts, 1) + env.rb

        # data points outside the target set
        states_not_ts = (env.state_space_low - env.lb) * torch.rand(batch_size_not_ts, 1) + env.lb

        # total data points
        states = torch.cat((states_ts, states_not_ts))
        actions = (env.action_space_low - env.action_space_high) * torch.rand(batch_size, 1) + env.action_space_high

        return states, actions

    # sample points
    states, actions = sample_data_points()

    fig, ax = plt.subplots()
    ax.scatter(states, actions)
    plt.show()

    # targets
    q_values_target_ts = torch.zeros((batch_size_ts, 1))
    q_values_target_not_ts = - torch.rand(batch_size_not_ts, 1)
    q_values_target = torch.cat((q_values_target_ts, q_values_target_not_ts))

    # schuffle
    idx = torch.randperm(batch_size)
    states = states[idx]
    q_values_target = q_values_target[idx]

    # compute q values
    q_values = critic.forward(states, actions)

    # train
    for i in range(n_iterations):

        if i % 10 == 0:
            pass

            # sample points
            #states, actions = sample_data_points()

        # compute q values
        q_values = critic.forward(states, actions)

        # compute mse loss
        loss = ((q_values - q_values_target)**2).mean()

        # compute gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(i, loss.item())

        if i % 1000 == 0:
            # compute tables following q-value model
            q_table, _, _, _ = compute_tables_critic(env, critic)
            plot_q_value_function(env, q_table)

def pre_train_critic(env, critic):

    optimizer = optim.Adam(critic.parameters(), lr=1e-6)

    # batch size
    batch_size_ts = 5 * 10**2
    batch_size_not_ts = 5 * 10**2
    batch_size = batch_size_ts + batch_size_not_ts

    # number of iterations
    n_iterations = 10**4

    env.action_space_low = 0
    env.action_space_high = 3

    def sample_data_points():

        # data points in the target set
        states_ts = (env.lb - env.rb) * torch.rand(batch_size_ts, 1) + env.rb

        # data points outside the target set
        states_not_ts = (env.state_space_low - env.lb) * torch.rand(batch_size_not_ts, 1) + env.lb

        # total data points
        states = torch.cat((states_ts, states_not_ts))
        actions = (env.action_space_low - env.action_space_high) * torch.rand(batch_size, 1) + env.action_space_high

        return states, actions

    # sample points
    states, actions = sample_data_points()

    fig, ax = plt.subplots()
    ax.scatter(states, actions)
    plt.show()

    # targets
    q_values_target_ts = torch.zeros((batch_size_ts, 1))
    q_values_target_not_ts = - torch.rand(batch_size_not_ts, 1)
    q_values_target = torch.cat((q_values_target_ts, q_values_target_not_ts))

    # schuffle
    idx = torch.randperm(batch_size)
    states = states[idx]
    q_values_target = q_values_target[idx]

    # compute q values
    q_values = critic.forward(states, actions)

    # train
    for i in range(n_iterations):

        if i % 10 == 0:
            pass

            # sample points
            #states, actions = sample_data_points()

        # compute q values
        q_values = critic.forward(states, actions)

        # compute mse loss
        loss = ((q_values - q_values_target)**2).mean()

        # compute gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(i, loss.item())

        if i % 1000 == 0:
            # compute tables following q-value model
            q_table, _, _, _ = compute_tables_critic(env, critic)
            plot_q_value_function(env, q_table)

