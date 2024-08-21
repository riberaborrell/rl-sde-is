import gymnasium as gym
import gym_sde_is

from rl_sde_is.dpg.reinforce_deterministic_core import reinforce_deterministic
from rl_sde_is.utils.base_parser import get_base_parser


def main():
    parser = get_base_parser()
    parser.description = 'Run model-based reinforce for deterministic policies for the sde \
                          importance sampling environment with the butane molecule.'
    args = parser.parse_args()

    # create gym environment
    env = gym.make(
        'sde-is-butane-{}-v0'.format(args.setting),
        temperature=args.temperature,
        gamma=10.0,
        T=args.T,
        state_init_dist=args.state_init_dist,
        is_torch=True,
    )

    # run reinforve algorithm with a deterministic policy
    data = reinforce_deterministic(
        env,
        gamma=args.gamma,
        n_layers=args.n_layers,
        d_hidden_layer=args.d_hidden,
        batch_size=args.batch_size,
        lr=args.lr,
        n_grad_iterations=args.n_grad_iterations,
        seed=args.seed,
        learn_value=args.learn_value,
        lr_value=args.lr_value,
        log_freq=args.log_freq,
        backup_freq=args.backup_freq,
        load=args.load,
    )

    # do plots
    if not args.plot:
        return

if __name__ == "__main__":
    main()
