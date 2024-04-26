#!/usr/bin/env python3

import numpy as np

from rl_sde_is.utils.path import load_data, save_data

def env(s, gym_env, args):


    # Initializing environment and random seed
    sampleId = s["Sample Id"]
    launchId = s["Launch Id"]
    _, _ = gym_env.reset(seed=sampleId * 1024 + launchId)

    # initial state
    s["State"] = gym_env.unwrapped._state.tolist()
    step = 0
    done = False

    while not done and step < args.n_steps_lim:

        # Getting new action
        s.update()

        # Performing the action
        action = np.array(s["Action"], dtype=np.float32)
        obs, r, done, _, info = gym_env.step(action)

        # Getting Reward
        s["Reward"] = r

        # Storing New State
        s["State"] = obs.tolist()

        # Advancing step counter
        step = step + 1

    # Setting termination status
    if done:
        s["Termination"] = "Terminal"
    else:
        s["Termination"] = "Truncated"

    # checkpoint results
    if gym_env.episode_count % args.backup_freq_episodes == 0:

        # collect results
        data = {}
        data['time_steps'] = gym_env.lengths
        data['returns'] = gym_env.returns
        data['log_psi_is'] = gym_env.log_psi_is

        # save results
        save_data(data, args.rel_dir_path)
