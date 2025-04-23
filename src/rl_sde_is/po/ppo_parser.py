def add_ppo_arguments(parser):
    parser.add_argument(
        '--n-mini-batches',
        type=int,
        default=32,
        help='the number of mini-batches',
    )
    parser.add_argument(
        '--anneal-lr',
        type=bool,
        default=True,
        help='toggle learning rate annealing for policy and value networks'
    )
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='the lambda for the general advantage estimation',
    )
    parser.add_argument(
        '--update-epochs',
        type=int,
        default=10,
        help='the K epochs to update the policy'
    )
    parser.add_argument(
        '--clip-coef',
        type=float,
        default=0.2,
        help='the surrogate clipping coefficient',
    )
    parser.add_argument(
        '--clip-vloss',
        type=bool,
        default=True,
        help='toggles whether or not to use a clipped loss for the value function, as per the paper',
    )
    parser.add_argument(
        '--ent-coef',
        type=float,
        default=0.0,
        help='coefficient of the entropy',
    )
    parser.add_argument(
        '--vf-coef',
        type=float,
        default=0.5,
        help='coefficient of the value function',
    )
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='the maximum norm for the gradient clipping',
    )
    parser.add_argument(
        '--target-kl',
        type=float,
        default=None,
        help='the target KL divergence threshold',
    )
    parser.add_argument(
        '--torch-deterministic',
        type=bool,
        default=True,
        help='if toggled, `torch.backends.cudnn.deterministic=False`',
    )
    return parser
