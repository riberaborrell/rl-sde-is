#!/usr/bin/env python3
import pickle

import numpy as np

# generate states file 'states.pickle'?
output = False


def environment(s, gym_env, args):


    # Initializing environment and random seed
    sampleId = s["Sample Id"]
    launchId = s["Launch Id"]
    _, _ = gym_env.reset(seed=sampleId * 1024 + launchId)

    # 
    d = gym_env.unwrapped.d
    gym_env.positions = np.zeros((d, args.max_steps))
    gym_env.actions = np.zeros((d, args.max_steps))
    s["State"] = gym_env.unwrapped._state.tolist()
    step = 0
    done = False

    while not done and step < args.max_steps:

        # Getting new action
        s.update()

        # Performing the action
        action = np.array(s["Action"], dtype=np.float32)
        obs, r, done, _, _ = gym_env.step(action)

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

    # Generate output file with states and actions
    if output:
        data = {
            'position': gym_env.positions,
            'action': gym_env.actions,
        }

        with open('states.pickle', 'wb') as fp:
            pickle.dump(data, fp)
