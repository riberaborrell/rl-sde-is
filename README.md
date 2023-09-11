# rl-sde-is

This Python repository contains the implementation of reinforcement learning algorithms for the importance sampling (IS) stochastic optimal control (SOC) environment. This environment represents the optimization problem assosiated to the importance sampling of metastable diffusion processes. In particularly, we aim to estimate path functionals up to a random time of stochastic process following an overdamped Langevin equation. We consider only first hitting time problems leading to an optimal, time-homogeneous control.

## Contains

- Planning methods in the tabular setting. We discretize the state and the action space and the state-action transition probability tensor and state-action reward are approximated.
- REINFORCE algorithm for deterministic policies. 
- TD3 episodic algorithm.

## Install

1. install the sde-hjb-solver package. Instructions can be found [here](https://github.com/riberaborrell/sde-hjb-solver)

2. clone the repo 
```
git clone git@github.com:riberaborrell/rl-sde-is.git
```

3. move inside the directory, create virtual environment and install required packages
```
cd rl-sde-is
make venv
```


4. activate venv
```
source venv/bin/activate
```

5. create config.py file and edit it
```
cp src/rl_sde_is/config_template.py src/rl_sde_is/config.py
```


## Developement

in step 2) also install developement packages
```
make develop
```

## Usage
1. Tabular methods

Create state-action-next-state transition probability tensor and state-action reward matrix after discretizing the state space and the action space:
```
$ python src/rl_sde_is/tabular_dp_tables.py --alpha 1 --beta 1 --dt 0.005 --h-state 0.1 --h-action 0.1 --plot
```

Run one q-value iteration algorithm:
```
$ python src/rl_sde_is/tabular_dp_qvalue_iteration.py --alpha 1 --beta 1 --dt 0.005 --h-state 0.1 --h-action 0.1 --n-iterations 2000 --live-plot
```


2. REINFORCE deterministic
Run the REINFORCE algorithm for deterministic policies.
```
$ python src/rl_sde_is/reinforce_deterministic_1d.py --alpha 1 --beta 1 --dt 0.005 --batch-size 100 --lr 1e-2 --n-iterations 1000 --seed 1 --backup-freq-iterations 100 --live-plot
```
Test the deterministic policy at each backed up iteration by computing the l2 error between the policy and the optimal policy along a batch of trajectories.
```
$ python src/rl_sde_is/reinforce_deterministic_1d.py --alpha 1 --beta 1 --dt 0.005 --batch-size 100 --lr 1e-2 --n-iterations 1000 --seed 1 --load --test --test-bath-size 1000 --test-freq-iterations 100 --plot
```


3. Deterministic policy gradient (TD3 variant)

Run the Twin Delayed Deep Deterministic Policy gradient (TD3)
```
$ python src/rl_sde_is/td3_episodic_1d.py --alpha 1 --beta 1 --dt 0.005 --n-steps-lim 1000 --n-episodes 10000 --expl-noise-init 1. --target-noise 0.1 --policy-delay 2 --action-limit 5 --seed 1 --live-plot --backup-freq-iterations 100
```
Test the deterministic policy at each backed up episode
```
$ python src/rl_sde_is/td3_episodic_1d.py --alpha 1 --beta 1 --dt 0.005 --n-steps-lim 1000 --n-episodes 10000 --expl-noise-init 1. --target-noise 0.1 --policy-delay 2 --action-limit 5 --seed 1 --load --test --test-bath-size 1000 --test-freq-episodes 100 --plot
```
