import argparse

def get_base_parser():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument(
        '--d',
        dest='d',
        type=int,
        default=1,
        help='Dimension of the environment. Default: 1',
    )
    parser.add_argument(
        '--alpha',
        dest='alpha',
        type=float,
        default=1.,
        help='Double well potential parameter. Default: 1.',
    )
    parser.add_argument(
        '--beta',
        dest='beta',
        type=float,
        default=1.,
        help='Set inverse of the temperature. Default: 1.',
    )
    parser.add_argument(
        '--dt',
        dest='dt',
        type=float,
        default=0.005,
        help='Set Euler-Marujama time discretization step. Default: 0.005',
    )
    parser.add_argument(
        '--seed',
        dest='seed',
        type=int,
        help='random seed. Default: None',
    )
    parser.add_argument(
        '--gamma',
        dest='gamma',
        type=float,
        default=1.,
        help='discount factor (default: 1.)',
    )
    parser.add_argument(
        '--s-init',
        dest='s_init',
        type=float,
        default=-1.,
        help='Set initial state of the trajectory. Default: -1.',
    )
    parser.add_argument(
        '--es',
        dest='explorable_starts',
        action='store_true',
        help='the initial point of the trajectory is uniform sampled in the domain \ target set.',
    )
    parser.add_argument(
        '--constant-lr',
        dest='constant_lr',
        action='store_true',
        help='the step size / learning rate parameter is constant.',
    )
    parser.add_argument(
        '--lr-type',
        dest='lr_type',
        choices=['constant', 'adaptive'],
        default='constant',
        help='Type of learning rate. Default: constant',
    )
    parser.add_argument(
        '--lr',
        dest='lr',
        type=float,
        default=0.01,
        help='Set learning rate. Default: 0.01',
    )
    parser.add_argument(
        '--lr-actor',
        dest='lr_actor',
        type=float,
        default=0.0001,
        help='Set learning rate for the actor network. Default: 0.0001',
    )
    parser.add_argument(
        '--lr-critic',
        dest='lr_critic',
        type=float,
        default=0.0001,
        help='Set learning rate for the critic network. Default: 0.0001',
    )
    parser.add_argument(
        '--lam',
        dest='lam',
        type=float,
        default=0.,
        help='Set lambda parameter for the lambda Sarsa algorithm. Default: 0.',
    )
    parser.add_argument(
        '--eps-type',
        dest='eps_type',
        choices=['constant', 'harmonic', 'linear-decay', 'exp-decay'],
        default='linear-decay',
        help='Type of epsilon succession. Default: constant',
    )
    parser.add_argument(
        '--eps-init',
        dest='eps_init',
        type=float,
        default=0.01,
        help='Set probility of picking an action randomly. Default: 0.01',
    )
    parser.add_argument(
        '--decay',
        dest='decay',
        type=float,
        default=1.,
        help='Set decay rate. Default: 1.',
    )
    parser.add_argument(
        '--eps-min',
        dest='eps_min',
        type=float,
        default=0.,
        help='Set minimum value for epsilon. Default: 0.0',
    )
    parser.add_argument(
        '--eps-max',
        dest='eps_max',
        type=float,
        default=1,
        help='Set maximum value for epsilon. Default: 1',
    )
    parser.add_argument(
        '--n-total-steps',
        dest='n_total_steps',
        type=int,
        default=10**6,
        help='Set number of maximum steps for the algorithm. Default: 1,000,000',
    )
    parser.add_argument(
        '--n-steps-lim',
        dest='n_steps_lim',
        type=int,
        default=10**6,
        help='Set number of maximum steps for an episode. Default: 1,000,000',
    )
    parser.add_argument(
        '--n-episodes',
        dest='n_episodes',
        type=int,
        default=1000,
        help='Set number of episodes. Default: 1000',
    )
    parser.add_argument(
        '--n-avg-episodes',
        dest='n_avg_episodes',
        type=int,
        default=100,
        help='Set number last episodes to averaged the statistics. Default: 100',
    )
    parser.add_argument(
        '--test-freq-episodes',
        dest='test_freq_episodes',
        type=int,
        default=10,
        help='Set frequency of model tests in terms of number of episodes. Default: 10',
    )
    parser.add_argument(
        '--backup-freq-episodes',
        dest='backup_freq_episodes',
        type=int,
        help='Set frequency of backups in terms of number of episodes. Default: None',
    )
    parser.add_argument(
        '--n-iterations',
        dest='n_iterations',
        type=int,
        default=1000,
        help='Set number of iterations. Default: 1000',
    )
    parser.add_argument(
        '--n-avg-iterations',
        dest='n_avg_iterations',
        type=int,
        default=10,
        help='Set number last iterations to averaged the statistics. Default: 10',
    )
    parser.add_argument(
        '--test-freq-iterations',
        dest='test_freq_iterations',
        type=int,
        default=10,
        help='Set frequency of model tests in terms of number of iterations. Default: 10',
    )
    parser.add_argument(
        '--backup-freq-iterations',
        dest='backup_freq_iterations',
        type=int,
        help='Set frequency of backups in terms of number of iterations . Default: None',
    )
    parser.add_argument(
        '--n-epochs',
        dest='n_epochs',
        type=int,
        default=100,
        help='Set number of epochs. Default: 100',
    )
    parser.add_argument(
        '--n-steps-per-epoch',
        dest='n_steps_per_epoch',
        type=int,
        default=1000,
        help='Set number of time steps per epoch. Default: 1000',
    )
    parser.add_argument(
        '--target-update-freq',
        dest='target_update_freq',
        type=int,
        default=100,
        help='Set number of time steps per epoch. Default: 1000',
    )
    parser.add_argument(
        '--step-sliced-episodes',
        dest='step_sliced_episodes',
        type=int,
        default=10,
        help='Set slice episodes step. Default: 10',
    )
    parser.add_argument(
        '--batch-size',
        dest='batch_size',
        type=int,
        default=1000,
        help='Set number of trajectories in each batch. Default: 1000',
    )
    parser.add_argument(
        '--test-batch-size',
        dest='test_batch_size',
        type=int,
        default=10,
        help='Set number of trajectories in each test batch. Default: 10',
    )
    parser.add_argument(
        '--replay-size',
        dest='replay_size',
        type=int,
        default=int(1e6),
        help='Set number of data slices in the replay buffer. Default: 10^6',
    )
    parser.add_argument(
        '--h-state',
        dest='h_state',
        type=float,
        default=0.1,
        help='Set the discretization step size for the state space. Default: 0.1',
    )
    parser.add_argument(
        '--h-action',
        dest='h_action',
        type=float,
        default=0.5,
        help='Set the discretization step size for the action space. Default: 0.5',
    )
    parser.add_argument(
        '--n-layers',
        dest='n_layers',
        type=int,
        default=3,
        help='Set total number of layers. Default: 3',
    )
    parser.add_argument(
        '--d-hidden-layer',
        dest='d_hidden_layer',
        type=int,
        default=32,
        help='Set dimension of the hidden layers. Default: 32',
    )
    parser.add_argument(
        '--expl-noise-init',
        dest='expl_noise_init',
        type=float,
        default=1.0,
        help='Set initial exploration noise. Default: 1.0',
    )
    parser.add_argument(
        '--policy-delay',
        dest='policy_delay',
        type=int,
        default=2,
        help='Set actor update frequency with respect to the critic update. Default: 2',
    )
    parser.add_argument(
        '--target-noise',
        dest='target_noise',
        type=float,
        default=0.2,
        help='Set target noise for policy smoothing. Default: 0.2',
    )
    parser.add_argument(
        '--action-limit',
        dest='action_limit',
        type=float,
        default=10,
        help='Set action limit for policy smoothing. Default: 10',
    )
    parser.add_argument(
        '--polyak',
        dest='polyak',
        type=float,
        default=0.995,
        help='Set polyak parameter for soft target updates. Default: 0.995',
    )
    parser.add_argument(
        '--dense',
        dest='is_dense',
        action='store_true',
        help='Flag determining if the NN Architecture is dense. Default: False',
    )
    parser.add_argument(
        '--return-estimator',
        dest='return_estimator',
        type=str,
        default='total-rewards',
        help='Type of return estimator. Default: "total rewards"',
    )
    parser.add_argument(
        '--test',
        dest='test',
        action='store_true',
        help='Test policy. Default: False',
    )
    parser.add_argument(
        '--load',
        dest='load',
        action='store_true',
        help='Load already run agent. Default: False',
    )
    parser.add_argument(
        '--plot',
        dest='plot',
        action='store_true',
        help='Plot resulted data. Default: False',
    )
    parser.add_argument(
        '--live-plot',
        dest='live_plot',
        action='store_true',
        help='Plot live results. Default: False',
    )
    parser.add_argument(
        '--plot-episode',
        dest='plot_episode',
        type=int,
        default=None,
        help='Episode that we want to plot. Default: None',
    )
    parser.add_argument(
        '--plot-iteration',
        dest='plot_iteration',
        type=int,
        default=None,
        help='Iteration that we want to plot. Default: None',
    )
    parser.add_argument(
        '--save-traj',
        dest='save_traj',
        action='store_true',
        help='Save states, actions and rewards of the first trajectory. Default: False',
    )
    return parser
