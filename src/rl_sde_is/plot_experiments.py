
from rl_sde_is.environments import DoubleWellStoppingTime1D
from rl_sde_is.reinforce_deterministic_vectorized import *
from rl_sde_is.plots import *

def main():

    # initialize environment
    env = DoubleWellStoppingTime1D(alpha=1., beta=1.)

    # set explorable starts flag
    env.is_state_init_sampled = False

    # set action space bounds
    env.action_space_low = 0
    env.action_space_high = 5

    # discretized state space (for plot purposes only)
    env.discretize_state_space(h_state=0.01)

    # get hjb solver
    sol_hjb = env.get_hjb_solver()

    # run reinforve algorithm with a deterministic policy
    data = reinforce(
        env,
        gamma=1.,
        batch_size=int(1e1),
        lr=1e-4,
        n_iterations=int(1e1),
        seed=1,
        load=True,
    )



if __name__ == "__main__":
    main()
