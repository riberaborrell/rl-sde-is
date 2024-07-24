#!/usr/bin/env python3

import numpy as np

from rl_sde_is.utils.path import load_data, save_data
from rl_sde_is.vracer.vracer_utils import collect_vracer_results

def env(s, gym_env, args):


    # Initializing environment and random seed
    sampleId = s["Sample Id"]
    launchId = s["Launch Id"]
    _, _ = gym_env.reset(seed=sampleId * 1024 + launchId)

    # initial state
    s["State"] = gym_env.unwrapped.state.tolist()

    done = False
    while not done:

        # Getting new action
        s.update()

        # Performing the action
        action = np.array(s["Action"], dtype=np.float32)
        obs, r, terminated, truncated, info = gym_env.step(action)

        # Getting Reward
        s["Reward"] = r

        # Storing New State
        s["State"] = obs.tolist()

        # interrupt if terminal state is reached or the episode is truncated
        done = terminated or truncated

    # Setting termination status
    s["Termination"] = "Terminal" if terminated else "Truncated"

    # checkpoint results
    if gym_env.episode_count % args.backup_freq == 0:

        # save results
        data = collect_vracer_results(gym_env)
        save_data(data, args.dir_path)
