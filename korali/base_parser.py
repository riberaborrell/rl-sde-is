import argparse

def get_base_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--state-init-dist',
        choices=['delta', 'uniform'],
        default='delta',
        help='state initial distribution.',
    )
    parser.add_argument(
        '--reward-type',
        choices=['state-action', 'state-action-next-state', 'baseline'],
        default='state-action',
        help='the step size / learning rate parameter is constant.',
    )
    parser.add_argument(
        '--distribution',
        help='Policy Distribution',
        type=str,
        default='Normal',
    )
    parser.add_argument(
        '--n-episodes',
        help='Number of episodes.',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--max-experiences',
        help='Number of experiences to collect.',
        type=int,
        default=int(1e5),
    )
    parser.add_argument(
        '--max-steps',
        help='Maximum number of steps for an episode.',
        type=int,
        default=int(1e5),
    )
    parser.add_argument(
        '--d-hidden',
        type=int,
        default=32,
        help='Set dimension of the hidden layers. Default: 32',
    )
    return parser
