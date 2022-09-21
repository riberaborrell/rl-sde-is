import numpy as np

from rl_sde_is.base_parser import get_base_parser
from rl_sde_is.environments import DoubleWellStoppingTime1D

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def main():
    args = get_parser().parse_args()

    # initialize environments
    env = DoubleWellStoppingTime1D(dt=args.dt)

    # set explorable starts flag
    if args.explorable_starts:
        env.is_state_init_sampled = True

    # reset environment
    state = env.reset()

    # terminal state flag
    complete = False

    for k in range(args.n_steps_lim):

        # interrupt if we are in a terminal state
        if complete:
            break

        # copy state
        state_copy = state.copy()

        # take a random action
        action = np.random.rand(1) * env.action_space_high

        # step dynamics forward
        new_state, r, complete = env.step(state, action)

        print('step: {}, state: {:.1f}, action: {:.1f}, reward: {:.3f}'
              ''.format(k, state_copy[0], action[0], r))

        # update state
        state = new_state


if __name__ == '__main__':
    main()
