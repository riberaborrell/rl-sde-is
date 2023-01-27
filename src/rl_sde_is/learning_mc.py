import gym
import gym_sde
import numpy as np
from sde.langevin_sde import LangevinSDE
from hjb.hjb_solver import SolverHJB

from sderl_tabular.agent_tabular import TabularAgent
from sderl_tabular.base_parser import get_base_parser
from sderl_tabular.utils_path import get_learning_mc_dir_path

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def main():
    args = get_parser().parse_args()

    # create gym env 
    if not args.explorable_starts:
        env = gym.make('sde-v0', beta=1., x_init=-1.)
    else:
        env = gym.make('sde-v0', beta=1., is_x_init_random=True)

    # initialize Agent
    agent = TabularAgent(
        env=env,
        name='learning-mc',
        gamma=1.,
        logs=args.do_report,
        constant_alpha=args.constant_alpha,
        alpha=args.alpha,
        n_episodes=args.n_episodes_lim,
    )

    # set state space and discretize
    agent.set_state_space()
    agent.discretize_state_space(h=args.h_state)
    agent.discretize_action_space(h=args.h_action, low=-3, high=3.)

    # set epsilons
    agent.set_epsilon_parameters(
        eps_type=args.eps_type,
        eps_init=args.eps_init,
        eps_min=args.eps_min,
        eps_max=args.eps_max,
        eps_decay=args.eps_decay,
    )

    # get dir path
    agent.set_env_dir_path()
    agent.set_dir_path()

    # run mc-learning agent
    if not args.load:

        # preallocate information for all epochs
        agent.preallocate_episodes()

        # set number of averaged episodes 
        agent.n_avg_episodes = args.n_avg_episodes

        # mc learning algorithm
        mc_learning(agent, args.n_steps_lim, args.constant_alpha, args.alpha)

        # save agent
        agent.save()

    # load already run agent
    else:
        if not agent.load():
            return

    # do plots
    if args.do_plots:

        # initialize Langevin sde
        sde = LangevinSDE(
            problem_name='langevin_stop-t',
            potential_name='nd_2well',
            d=1,
            alpha=np.ones(1),
            beta=1.,
            domain=np.full((1, 2), [-2, 2]),
        )

        # initialize hjb solver
        h_hjb = 0.01
        sol_hjb = SolverHJB(sde, h=h_hjb)

        # load already computed solution
        sol_hjb.load()

        # discretization step ratio
        k = int(args.h_state / h_hjb)
        assert agent.state_space_h.shape == sol_hjb.u_opt[::k].shape, ''

        agent.episodes = np.arange(agent.n_episodes)
        agent.plot_total_rewards()
        agent.plot_time_steps()
        agent.plot_epsilons()
        agent.plot_frequency_table()
        agent.plot_q_table()
        agent.plot_greedy_policy(control_hjb=sol_hjb.u_opt[::k])
        agent.episodes = np.arange(agent.n_episodes)
        agent.plot_total_rewards()
        agent.plot_time_steps()
        agent.plot_epsilons()
        agent.plot_frequency_table()
        agent.plot_q_table()
        #agent.plot_sliced_q_tables()

    # print running avg if load
    if not args.load:
        return

    episodes = np.arange(agent.n_episodes)
    for ep in episodes:

        if args.do_report and ep % args.n_avg_episodes == 0:
            msg = agent.log_episodes(ep)
            print(msg)

def mc_learning(agent, n_steps_lim, constant_alpha=False, alpha=None):

    # initialize frequency and q-values table
    agent.initialize_frequency_table()
    agent.initialize_q_table()

    # for each episode
    for ep in np.arange(agent.n_episodes):

        # reset environment
        state = agent.env.reset()

        # reset trajectory
        agent.reset_trajectory()

        # terminal state flag
        complete = False

        # sample episode
        for k in np.arange(n_steps_lim):

            # interrupt if we are in a terminal state
            if complete:
                break

            # get index of the state
            idx_state = agent.get_state_idx(state)

            # choose action following epsilon greedy policy
            idx_action, action = agent.get_epsilon_greedy_action(ep, idx_state)

            # step dynamics forward
            new_state, r, complete, _ = agent.env.step(action)

            # save state, actions and reward
            agent.save_trajectory(state, action, r)

            # update state
            state = new_state

        # compute return
        agent.compute_discounted_rewards()
        agent.compute_returns()

        # update q values
        n_steps_trajectory = agent.states.shape[0]
        for k in np.arange(n_steps_trajectory):

            # state and its index at step k
            state = agent.states[k]
            idx_state = agent.get_state_idx(state)

            # action and its index at step k
            action = agent.actions[k]
            idx_action = agent.get_action_idx(action)

            # state-action index
            idx = (idx_state, idx_action)
            g = agent.returns[k]

            # update frequency table
            agent.n_table[idx] += 1

            # set learning rate
            if not constant_alpha:
                alpha = 1 / agent.n_table[idx]

            # update q table
            agent.q_table[idx] += alpha * (g - agent.q_table[idx])

        # save time steps
        agent.save_episode(ep, k)

        # logs
        if agent.logs and ep % agent.n_avg_episodes == 0:
            msg = agent.log_episodes(ep)
            print(msg)


    # update npz dict
    agent.update_npz_dict_agent()

    # save frequency and q-value last tables
    agent.save_frequency_table()
    agent.save_q_table()

if __name__ == '__main__':
    main()
