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
        choices=['brownian-1d', 'doublewell-1d', 'doublewell-2d',
                 'doublewell-nd', 'doublewell-nd-asym', 'triplewell', 'butane'],
        default='doublewell-1d',
        help='Set setting type. Default: doublewell-1d',
    )
    parser.add_argument(
        '--agent-type',
        choices=['random', 'uncontrolled', 'hjb', 'trained'],
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
        '--alpha-i',
        type=float,
        default=1.,
        help='Set i-th component of the barrier height parameter of the given potential. Default: [1.]',
    )
    parser.add_argument(
        '--alpha-j',
        type=float,
        default=1.,
        help='Set i-th component of the barrier height parameter of the given potential. Default: [1.]',
    )
    parser.add_argument(
        '--alpha-k',
        type=float,
        default=1.,
        help='Set i-th component of the barrier height parameter of the given potential. Default: [1.]',
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
        '--return-type',
        choices=['initial-return', 'n-return'],
        default='initial-return',
        help='Set type of return used. Default: initial-return',
    )
    parser.add_argument(
        '--expectation-type',
        choices=['random-time', 'on-policy', 'off-policy'],
        default='random-time',
        help='Set type of expectation. Default: random-time',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Set number of trajectories in each batch. Default: 1000',
    )
    parser.add_argument(
        '--mini-batch-size',
        type=int,
        default=None,
        help='Set mini batch size for on-policy expectations. Default: None',
    )
    parser.add_argument(
        '--mini-batch-size-type',
        choices=['constant', 'adaptive'],
        default='constant',
        help='Set type of mini batch size. Constant or adaptive relative to the \
              memory size. Default: constant',
    )
    parser.add_argument(
        '--estimate-z',
        action='store_true',
        help='Estimate the z normalization factor for the spg or dpg gradients.',
    )
    parser.add_argument(
        '--optim-type',
        choices=['sgd', 'adam'],
        default='adam',
        help='Set optimization routine. Default: adam',
    )
    parser.add_argument(
        '--constant-lr',
        action='store_true',
        help='the step size / learning rate parameter is constant.',
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
        '--learn-value',
        action='store_true',
        help='Learn the value function by minimizing the Bellman Loss.',
    )
    parser.add_argument(
        '--lr-value',
        type=float,
        default=0.001,
        help='Set learning rate for the value function network. Default: 0.001',
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
        '--n-total-steps',
        type=int,
        default=10**8,
        help='Set number of maximum steps for the algorithm. Default: 100,000,000',
    )
    parser.add_argument(
        '--n-steps-lim',
        type=int,
        default=10**6,
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
        '--n-grad-iterations',
        type=int,
        default=1000,
        help='Set number of gradient iterations. Default: 1000',
    )
    parser.add_argument(
        '--n-steps-per-epoch',
        type=int,
        default=1000,
        help='Set number of time steps per epoch. Default: 1000',
    )
    parser.add_argument(
        '--eval-freq',
        type=int,
        default=10,
        help='Set frequency of model evaluation. Default: 10',
    )
    parser.add_argument(
        '--backup-freq',
        type=int,
        default=100,
        help='Set frequency of backups. Default: 10',
    )
    parser.add_argument(
        '--update-freq',
        type=int,
        default=100,
        help='Set gradient update frequency in time steps. Default: 1000',
    )
    parser.add_argument(
        '--step-sliced-episodes',
        type=int,
        default=10,
        help='Set slice episodes step. Default: 10',
    )
    parser.add_argument(
        '--learning-starts',
        type=int,
        default=1000,
        help='Time step to start learning. Default: 1000',
    )
    parser.add_argument(
        '--eval-batch-size',
        type=int,
        default=10,
        help='Set number of trajectories in each model evaluation. Default: 10',
    )
    parser.add_argument(
        '--replay-size',
        type=int,
        default=100000,
        help='Set number of data slices in the replay memory. Default: 10^5',
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
        default=2,
        help='Set total number of layers. Default: 2',
    )
    parser.add_argument(
        '--d-hidden',
        type=int,
        default=32,
        help='Set dimension of the hidden layers. Default: 32',
    )
    parser.add_argument(
        '--theta-init',
        type=str,
        default='null',
        choices=['null', 'hjb'],
        help='Set if model parameters are trained to get the hjb solution. Default: null',
    )
    parser.add_argument(
        '--policy-type',
        type=str,
        default='det',
        choices=['det', 'stoch', 'stoch-mean'],
        help='Set type of policy. Used for evaluating the model. Default: det',
    )
    parser.add_argument(
        '--gaussian-policy-type',
        type=str,
        default='const-cov',
        choices=['const-cov', 'scheduled', 'learnt-cov'],
        help='Set if the covariance of the stochastic gaussian policy is constant, scheduled, or learnt. Default: const-cov',
    )
    parser.add_argument(
        '--policy-noise',
        type=float,
        default=1.0,
        help='Set factor of scalar covariance matrix. Default: 1.',
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
        default=None,
        help='Set action limit. Default: None',
    )
    parser.add_argument(
        '--polyak',
        type=float,
        default=0.995,
        help='Set polyak parameter for soft target updates. Default: 0.995',
    )
    parser.add_argument(
        '--cutoff-scale',
        type=float,
        default=4.0,
        help='Set cut-off scale for remember and forget algorithms. Default: 4.0',
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
        '--log-freq',
        type=int,
        default=100,
        help='Set frequency to logging algorithm results. Default: 1',
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
        '--live-plot-freq',
        type=int,
        default=None,
        help='Set frequency to live plots. Default: None',
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
