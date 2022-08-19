import argparse

def get_base_parser():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument(
        '--seed',
        dest='seed',
        type=int,
        default=1,
        help='random seed (default: 1)',
    )
    parser.add_argument(
        '--gamma',
        dest='gamma',
        type=float,
        default=1.,
        help='discount factor (default: 1.)',
    )
    parser.add_argument(
        '--x-init',
        dest='x_init',
        type=float,
        default=-1.,
        help='Set initial position of the trajectory. Default: -1.',
    )
    parser.add_argument(
        '--es',
        dest='explorable_starts',
        action='store_true',
        help='the initial point of the trajectory is uniform sampled.',
    )
    parser.add_argument(
        '--constant-alpha',
        dest='constant_alpha',
        action='store_true',
        help='the step size / learning rate parameter is given by alpha.',
    )
    parser.add_argument(
        '--alpha',
        dest='alpha',
        type=float,
        default=0.01,
        help='Set learning rate. Default: 0.01',
    )
    parser.add_argument(
        '--lr-actor',
        dest='lr_actor',
        type=float,
        default=0.01,
        help='Set learning rate for the actor network. Default: 0.01',
    )
    parser.add_argument(
        '--lr-critic',
        dest='lr_critic',
        type=float,
        default=0.01,
        help='Set learning rate for the critic network. Default: 0.01',
    )
    parser.add_argument(
        '--lam',
        dest='lam',
        type=float,
        default=0.5,
        help='Set lambda parameter for the lambda Sarsa algorithm. Default: 0.5',
    )
    parser.add_argument(
        '--eps-type',
        dest='eps_type',
        choices=['constant', 'harmonic', 'exp'],
        default='constant',
        help='Type of epsilon succession. Default: constant',
    )
    parser.add_argument(
        '--eps-init',
        dest='eps_init',
        type=float,
        default=0.5,
        help='Set probility of picking an action randomly. Default: 0.5',
    )
    parser.add_argument(
        '--eps-decay',
        dest='eps_decay',
        type=float,
        default=0.98,
        help='Set decay rate of epsilon. Default: 0.98',
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
        '--n-steps-lim',
        dest='n_steps_lim',
        type=int,
        default=10**6,
        help='Set number of maximum steps for an episode. Default: 1000',
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
        '--n-epochs',
        dest='n_epochs',
        type=int,
        default=100,
        help='Set number of epochs. Default: 100',
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
        default=10,
        help='Set number of trajectories in each batch. Default: 10',
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
        '--render',
        dest='do_render',
        action='store_true',
        help='render the environment',
    )
    parser.add_argument(
        '--log-interval',
        dest='log_interval',
        type=int,
        default=10,
        help='interval between training status logs (default: 10)',
    )
    parser.add_argument(
        '--load',
        dest='load',
        action='store_true',
        help='Load already run agent. Default: False',
    )
    parser.add_argument(
        '--do-plots',
        dest='do_plots',
        action='store_true',
        help='Do plots. Default: False',
    )
    parser.add_argument(
        '--do-report',
        dest='do_report',
        action='store_true',
        help='Write report. Default: False',
    )
    return parser
