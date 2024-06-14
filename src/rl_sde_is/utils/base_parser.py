import argparse

def get_base_parser():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument(
        '--setting',
        choices=['mgf', 'committor', 'fht-prob'],
        default='mgf',
        help='Set setting type. Default: mgf',
    )
    parser.add_argument(
        '--problem',
        choices=['doublewell-1d', 'doublewell-2d', 'triplewell', 'butane'],
        default='doublewell-1d',
        help='Set setting type. Default: doublewell-1d',
    )
    parser.add_argument(
        '--agent-type',
        choices=['random', 'not-controlled', 'hjb', 'trained'],
        default='random',
        help='Type of agent. Default: random',
    )
    parser.add_argument(
        '--n-envs',
        type=int,
        default=1,
        help='Set number of gym environments. Default: 1',
    )
    parser.add_argument(
        '--state-init-dist',
        choices=['delta', 'uniform'],
        default='delta',
        help='Set state initial distribution.',
    )
    parser.add_argument(
        '--reward-type',
        choices=['state-action', 'state-action-next-state', 'baseline'],
        default='state-action',
        help='the step size / learning rate parameter is constant.',
    )
    parser.add_argument(
        '--baseline-scale-factor',
        type=float,
        default=1.,
        help='Scaling factor of the added reward baseline. Default: 1.',
    )
    parser.add_argument(
        '--d',
        type=int,
        default=1,
        help='Dimension of the environment. Default: 1',
    )
    parser.add_argument(
        '--alpha',
        type=float,
        nargs='+',
        default=[1.],
        help='Potential barrier parameter. Default: 1.',
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=1.,
        help='Set inverse of the temperature. Default: 1.',
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=600.,
        help='Set temperature. Default: 600.',
    )
    parser.add_argument(
        '--dt',
        type=float,
        default=0.005,
        help='Set Euler-Marujama time discretization step. Default: 0.005',
    )
    parser.add_argument(
        '--T',
        type=float,
        default=1.0,
        help='Set finite-time horizon. Default: 1.0',
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='random seed. Default: None',
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=1.,
        help='discount factor (default: 1.)',
    )
    parser.add_argument(
        '--s-init',
        type=float,
        default=-1.,
        help='Set initial state of the trajectory. Default: -1.',
    )
    parser.add_argument(
        '--constant-lr',
        action='store_true',
        help='the step size / learning rate parameter is constant.',
    )
    parser.add_argument(
        '--lr-type',
        choices=['constant', 'adaptive'],
        default='constant',
        help='Type of learning rate. Default: constant',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help='Set learning rate. Default: 0.01',
    )
    parser.add_argument(
        '--lr-actor',
        type=float,
        default=0.0001,
        help='Set learning rate for the actor network. Default: 0.0001',
    )
    parser.add_argument(
        '--lr-critic',
        type=float,
        default=0.0001,
        help='Set learning rate for the critic network. Default: 0.0001',
    )
    parser.add_argument(
        '--lam',
        type=float,
        default=0.,
        help='Set lambda parameter for the lambda Sarsa algorithm. Default: 0.',
    )
    parser.add_argument(
        '--eps-type',
        choices=['constant', 'harmonic', 'linear-decay', 'exp-decay'],
        default='linear-decay',
        help='Type of epsilon succession. Default: constant',
    )
    parser.add_argument(
        '--eps-init',
        type=float,
        default=0.01,
        help='Set probility of picking an action randomly. Default: 0.01',
    )
    parser.add_argument(
        '--decay',
        type=float,
        default=1.,
        help='Set decay rate. Default: 1.',
    )
    parser.add_argument(
        '--eps-min',
        type=float,
        default=0.,
        help='Set minimum value for epsilon. Default: 0.0',
    )
    parser.add_argument(
        '--eps-max',
        type=float,
        default=1,
        help='Set maximum value for epsilon. Default: 1',
    )
    parser.add_argument(
        '--n-total-steps',
        type=int,
        default=10**6,
        help='Set number of maximum steps for the algorithm. Default: 1,000,000',
    )
    parser.add_argument(
        '--n-steps-lim',
        type=int,
        default=int(1e6),
        help='Set number of maximum steps for an episode. Default: 1,000,000',
    )
    parser.add_argument(
        '--n-episodes',
        type=int,
        default=1000,
        help='Set number of episodes. Default: 1000',
    )
    parser.add_argument(
        '--n-avg-episodes',
        type=int,
        default=100,
        help='Set number last episodes to averaged the statistics. Default: 100',
    )
    parser.add_argument(
        '--test-freq-episodes',
        type=int,
        default=10,
        help='Set frequency of model tests in terms of number of episodes. Default: 10',
    )
    parser.add_argument(
        '--backup-freq-episodes',
        type=int,
        default=100,
        help='Set frequency of backups in terms of number of episodes. Default: 10',
    )
    parser.add_argument(
        '--n-iterations',
        type=int,
        default=1000,
        help='Set number of iterations. Default: 1000',
    )
    parser.add_argument(
        '--n-avg-iterations',
        type=int,
        default=10,
        help='Set number last iterations to averaged the statistics. Default: 10',
    )
    parser.add_argument(
        '--test-freq-iterations',
        type=int,
        default=10,
        help='Set frequency of model tests in terms of number of iterations. Default: 10',
    )
    parser.add_argument(
        '--backup-freq-iterations',
        type=int,
        help='Set frequency of backups in terms of number of iterations . Default: None',
    )
    parser.add_argument(
        '--n-epochs',
        type=int,
        default=100,
        help='Set number of epochs. Default: 100',
    )
    parser.add_argument(
        '--n-steps-per-epoch',
        type=int,
        default=1000,
        help='Set number of time steps per epoch. Default: 1000',
    )
    parser.add_argument(
        '--target-update-freq',
        type=int,
        default=100,
        help='Set number of time steps per epoch. Default: 1000',
    )
    parser.add_argument(
        '--step-sliced-episodes',
        type=int,
        default=10,
        help='Set slice episodes step. Default: 10',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Set number of trajectories in each batch. Default: 1000',
    )
    parser.add_argument(
        '--test-batch-size',
        type=int,
        default=10,
        help='Set number of trajectories in each test batch. Default: 10',
    )
    parser.add_argument(
        '--replay-size',
        type=int,
        default=int(1e6),
        help='Set number of data slices in the replay buffer. Default: 10^6',
    )
    parser.add_argument(
        '--h-state',
        type=float,
        default=0.1,
        help='Set the discretization step size for the state space. Default: 0.1',
    )
    parser.add_argument(
        '--h-action',
        type=float,
        default=0.5,
        help='Set the discretization step size for the action space. Default: 0.5',
    )
    parser.add_argument(
        '--n-layers',
        type=int,
        default=3,
        help='Set total number of layers. Default: 3',
    )
    parser.add_argument(
        '--d-hidden',
        type=int,
        default=32,
        help='Set dimension of the hidden layers. Default: 32',
    )
    parser.add_argument(
        '--expl-noise-init',
        type=float,
        default=1.0,
        help='Set initial exploration noise. Default: 1.0',
    )
    parser.add_argument(
        '--policy-freq',
        type=int,
        default=2,
        help='Set actor update frequency with respect to the critic update i.e. time steps. Default: 2',
    )
    parser.add_argument(
        '--target-noise',
        type=float,
        default=0.2,
        help='Set target noise for policy smoothing. Default: 0.2',
    )
    parser.add_argument(
        '--action-limit',
        type=float,
        default=10,
        help='Set action limit. Default: 10',
    )
    parser.add_argument(
        '--polyak',
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
        type=str,
        default='total-rewards',
        help='Type of return estimator. Default: "total rewards"',
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test policy. Default: False',
    )
    parser.add_argument(
        '--load',
        action='store_true',
        help='Load already run agent. Default: False',
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Plot resulted data. Default: False',
    )
    parser.add_argument(
        '--live-plot',
        action='store_true',
        help='Plot live results. Default: False',
    )
    parser.add_argument(
        '--plot-episode',
        type=int,
        default=None,
        help='Episode that we want to plot. Default: None',
    )
    parser.add_argument(
        '--plot-iteration',
        type=int,
        default=None,
        help='Iteration that we want to plot. Default: None',
    )
    parser.add_argument(
        '--save-traj',
        action='store_true',
        help='Save states, actions and rewards of the first trajectory. Default: False',
    )
    parser.add_argument(
        '--track-l2-error',
        action='store_true',
        help='track policy l2 error',
    )
    parser.add_argument(
        '--track',
        action='store_true',
        help='track gym environment with wandb',
    )
    return parser
