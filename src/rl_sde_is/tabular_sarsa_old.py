import gym
import gym_sde
import numpy as np

from sderl_tabular.sde_agent import TabularAgent
from sderl_tabular.base_parser import get_base_parser
from sderl_tabular.utils_path import get_sarsa_lambda_dir_path, get_eps_constant_dir_path, \
                                     get_eps_harmonic_dir_path

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
    agent = TabularAgent(env=env, gamma=1., logs=args.do_report)

    # get dir path
    agent.set_dir_path()

    if args.eps_type == 'constant':
        eps_rel_path = get_eps_constant_dir_path(args.eps_init)
    elif args.eps_type == 'harmonic':
        eps_rel_path = get_eps_harmonic_dir_path()
    else:
        return

    agent.dir_path = get_sarsa_lambda_dir_path(agent.dir_path, args.explorable_starts,
                                               args.h_state, args.h_action, args.alpha,
                                               args.lam, eps_rel_path, args.n_episodes_lim)

    # run sarsa lambda agent
    if not args.load:

        # preallocate information for all epochs
        agent.n_episodes = args.n_episodes_lim
        agent.preallocate_episodes()

        # set number of averaged episodes 
        agent.n_avg_episodes = args.n_avg_episodes

        # set state space and discretize
        agent.set_state_space()
        agent.discretize_state_space(h=args.h_state)
        agent.discretize_action_space(h=args.h_action)

        # set epsilons
        if args.eps_type == 'constant':
            agent.set_epsilons_constant(args.eps_init)
        elif args.eps_type == 'harmonic':
            agent.set_epsilons_harmonic()

        # sarsa lambda algorithm
        sarsa_lambda(agent, args.n_steps_lim, args.alpha, args.lam)

        # save agent
        agent.save()

    # load already run agent
    else:
        if not agent.load():
            return

        # set state space and discretize
        agent.set_state_space()
        agent.discretize_state_space(h=args.h_state)
        agent.discretize_action_space(h=args.h_action)

    # do plots
    if args.do_plots:
        agent.episodes = np.arange(agent.n_episodes)
        agent.plot_total_rewards()
        agent.plot_time_steps()
        agent.plot_epsilons()
        agent.plot_frequency_table()
        agent.plot_q_table()
        agent.plot_control()
        #agent.plot_sliced_q_tables()


    # print running avg if load
    if not args.load:
        return

    episodes = np.arange(agent.n_episodes)
    for ep in episodes:

        if args.do_report and ep % args.n_avg_episodes == 0:
            msg = agent.log_episodes(ep)
            print(msg)

def sarsa_lambda(agent, n_steps_lim, alpha, lam):

    # initialize q-values table
    agent.initialize_frequency_table()
    agent.initialize_q_table()
    #agent.initialize_eligibility_traces()

    # for each episode
    for ep in np.arange(agent.n_episodes):

        # reset environment and choose action
        state = agent.env.reset()
        idx_state = agent.get_state_idx(state)
        idx_action, action = agent.get_epsilon_greedy_action(ep, idx_state)
        idx = (idx_state, idx_action,)

        # reset rewards
        agent.reset_rewards()

        # terminal state flag
        complete = False

        # sample episode
        for k in np.arange(n_steps_lim):

            # interrupt if we are in a terminal state
            if complete:
                break

            # step dynamics forward
            new_state, r, complete, _ = agent.env.step(action)
            idx_new_state = agent.get_state_idx(new_state)

            # get new action
            idx_new_action, new_action = agent.get_epsilon_greedy_action(ep, idx_new_state)
            idx_new = (idx_new_state, idx_new_action,)

            # update frequency table
            agent.n_table[idx] += 1

            # compute temporal difference error
            td_error = r + agent.gamma * agent.q_table[idx_new] - agent.q_table[idx]

            # update eligibility traces table
            #agent.e_table[idx] += 1

            # update the whole q-value and eligibility traces tables
            agent.q_table = agent.q_table + alpha * td_error# * agent.e_table
            #agent.e_table = agent.e_table * agent.gamma * lam

            # save reward
            agent.save_reward(r)

            # update state and action
            idx = idx_new
            state = new_state
            action = new_action

        # compute return
        agent.compute_discounted_rewards()
        agent.compute_returns()

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
