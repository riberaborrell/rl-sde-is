#!/usr/bin/env python3

import numpy as np

def environment(s, gym_env, args):


    # Initializing environment and random seed
    sampleId = s["Sample Id"]
    launchId = s["Launch Id"]
    _, _ = gym_env.reset(seed=sampleId * 1024 + launchId)

    # dimension of the state and action space
    d = gym_env.unwrapped.d
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
