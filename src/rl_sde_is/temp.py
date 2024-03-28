# initial distribution depends on the iteration

# initialize trajectories
states = torch.FloatTensor(env.reset_moving_normal(it, batch_size=K))

def reset_moving_normal(self, it, batch_size=1):
    x_a, x_b = -1., 1
    mu = it * (x_a - x_b) / 10**3 + x_b
    sigma = 1
    return np.expand_dims(np.random.normal(mu, sigma, batch_size), axis=1)


# old method for choosing the epsilons for tabular off policy learning
def set_epsilons(eps_type, **kwargs):

    # constant sequence
    if eps_type == 'constant':
        epsilons = get_epsilons_constant(
            n_episodes=kwargs['n_episodes'],
            eps_init=kwargs['eps_init'],
        )

    # linear decaying sequence
    elif eps_type == 'linear-decay':
        epsilons = get_epsilons_linear_decay(
            n_episodes=kwargs['n_episodes'],
            eps_min=kwargs['eps_min'],
        )

    # exponential decaying sequence
    elif eps_type == 'exp-decay':
        epsilons = get_epsilons_exp_decay(
            n_episodes=kwargs['n_episodes'],
            eps_init=kwargs['eps_init'],
            eps_decay=kwargs['eps_decay']
        )

    # harmonic decreasing sequence
    elif eps_type == 'harmonic':
        epsilons = get_epsilons_harmonic(
            n_episodes=kwargs['n_episodes'],
        )

    return epsilons

# old advantage learning update. Reference?
a_table[idx] = np.max(a_table[idx_state]) \
    + lr * (
    r \
    + gamma * np.max(a_table[idx_new_state]) \
    - np.max(a_table[idx])
)

    env.action_space_bounds[0] = 0
    env.action_space_bounds[1] = 5



    n_episodes = data['n_episodes']
    #step = data['test_freq_episodes']
    step = int(1e4)
    episodes = np.arange(0, n_episodes + step, step)
    canvas_det_policy_2d_figure(env, data, episodes, policy_opt, scale=5., width=0.003)
    return

