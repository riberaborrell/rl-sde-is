import numpy as np
import scipy.stats as stats
import torch

class DoubleWellCommittor1D():

    def __init__(self, beta=1., alpha=1., dt=0.005, epsilon=1e-10, is_state_init_sampled=False):

        # environment log name
        self.name = 'doublewell-1d-committor__beta{:.1f}_alpha{:.1f}'.format(beta, alpha)

        # double well potential
        self.d = 1
        self.alpha = alpha

        # sde
        self.beta = beta
        self.sigma = np.sqrt(2. / self.beta)
        self.sigma_tensor = torch.tensor(self.sigma, dtype=torch.float32)

        # Euler-Maruyama
        self.dt = dt
        self.dt_tensor = torch.tensor(self.dt, dtype=torch.float32)

        # committor
        self.epsilon = epsilon

        # target set
        self.target_set_a = (-2, -1)
        self.target_set_b = (1, 2)

        # initial state
        self.is_state_init_sampled = is_state_init_sampled
        self.state_init = np.zeros((1, self.d), dtype=np.float32)

        # observation space bounds
        self.state_space_dim = 1
        self.state_space_low = -2.
        self.state_space_high = 2.

        # action space bounds
        self.action_space_dim = 1
        self.action_space_low = 0.
        self.action_space_high = 3.

    def potential(self, state):
        return self.alpha * (state**2 - 1) ** 2

    def gradient(self, state):
        return 4 * self.alpha * state * (state**2 - 1)

    def is_done(self, state):
        done = np.where(
            (state[:, 0] <= self.target_set_a[1]) | (state[:, 0] >= self.target_set_b[0]),
            True,
            False,
        )
        return done

    def is_done_torch(self, state):
        done = torch.where(
            (state[:, 0] <= self.target_set_a[1]) | (state[:, 0] >= self.target_set_b[0]),
            True,
            False,
        )
        return done

    #def is_in_b(self, state):
    #    return np.where(state[:, 0] >= self.b_lb, True, False)
        #return np.where(state[:, 0] <= self.a_rb, True, False)

    #def is_in_b_torch(self, state):
    #    return torch.where(state[:, 0] >= self.b_lb, True, False)

    def f(self, state):
        return np.zeros(state.shape[0])

    def g(self, state):
        #return np.where(self.is_in_b(state), 0, 10**5)
        return np.where(
            (state >= self.target_set_b[0]) & (state <= self.target_set_b[1]),
            0,
            -np.log(self.epsilon),
        ).squeeze()

    def f_torch(self, state):
        return torch.zeros(state.shape[0])

    def g_torch(self, state):
        return torch.where(self.is_in_b_torch(state), 0, 10**5)

    def reset(self, batch_size=1):
        if not self.is_state_init_sampled:
            return np.full((batch_size, self.d), self.state_init)
        else:
            return np.full(
                (batch_size, self.d),
                np.random.uniform(self.target_set_a[1], self.target_set_b[0], (self.d,))
            )

    def state_action_transition_function(self, next_state, state, action, h):
        mu = state + (- self.gradient(state) + self.sigma * action) * self.dt
        std_dev = self.sigma * np.sqrt(self.dt)
        prob = stats.norm.cdf(next_state + h, mu, std_dev) \
             - stats.norm.cdf(next_state - h, mu, std_dev)
        return prob

    def reward_signal_state_action(self, state, action):
        done = self.is_done(state)
        reward = np.where(
            done,
            - self.g(state),
            - (self.f(state) + 0.5 * np.linalg.norm(action, axis=1)**2) * self.dt,
        )
        return reward, done

    def reward_signal_state_action_next_state(self, state, action, next_state):
        done = self.is_done(next_state)
        reward = np.where(
            done,
            - (self.f(state) + 0.5 * np.linalg.norm(action, axis=1)**2) * self.dt \
            - self.g(next_state),
            - (self.f(state) + 0.5 * np.linalg.norm(action, axis=1)**2) * self.dt,
        )
        return reward, done

    def reward_signal_state_action_torch(self, state, action):
        done = self.is_done_torch(state)
        reward = torch.where(
            done,
            - self.g_torch(state),
            - (self.f_torch(state) + 0.5 * torch.linalg.norm(action, axis=1)**2) * self.dt_tensor,
        )
        return reward, done

    def reward_signal_state_action_next_state_torch(self, state, action, next_state):
        done = self.is_done_torch(next_state)
        reward = torch.where(
            done,
            - (self.f_torch(state) + 0.5 * torch.linalg.norm(action, axis=1)**2) * self.dt_tensor \
            - self.g_torch(next_state),
            - (self.f_torch(state) + 0.5 * torch.linalg.norm(action, axis=1)**2) * self.dt_tensor,
        )
        return reward, done


    def step(self, state, action):

        # batch_size
        batch_size = state.shape[0]

        # brownian increment
        dbt = np.array(np.sqrt(self.dt) * np.random.randn(batch_size, self.d), dtype=np.float32)

        # sde step
        next_state = state \
                   + (- self.gradient(state) + self.sigma * action) * self.dt \
                   + self.sigma * dbt

        # reward signal r_{n+1} = r(s_{n+1}, s_n, a_n)
        r, done = self.reward_signal_state_action_next_state(state, action, next_state)

        # reward signal r_n = r(s_n, a_n)
        #r, done = self.reward_signal_state_action(state, action)

        return next_state, r, done, dbt

    def step_torch(self, state, action):

        # batch_size
        batch_size = state.shape[0]

        # brownian increment
        dt = self.dt_tensor
        dbt = torch.sqrt(dt) * torch.randn((batch_size, self.d), dtype=torch.float32)

        # sde step
        sigma = self.sigma_tensor
        next_state = state \
                   + (- self.gradient(state) + sigma * action) * dt \
                   + sigma * dbt

        # reward signal r_{n+1} = r(s_{n+1}, s_n, a_n)
        r, done = self.reward_signal_state_action_next_state_torch(state, action, next_state)

        # reward signal r_n = r(s_n, a_n)
        #r, done = self.reward_signal_state_action_torch(state, action)

        return next_state, r, done, dbt

    def get_idx_new_in_ts(self, is_in_target_set, been_in_target_set):

        idx = np.where(
                (is_in_target_set == True) &
                (been_in_target_set == False)
        )[0]

        been_in_target_set[idx] = True

        return idx

    def get_idx_new_in_ts_torch(self, is_in_target_set, been_in_target_set):

        idx = torch.where(
                (is_in_target_set == True) &
                (been_in_target_set == False)
        )[0]

        been_in_target_set[idx] = True

        return idx

    def discretize_state_space(self, h_state):

        # discretize state space
        self.state_space_h = np.around(
            np.arange(
                self.state_space_low,
                self.state_space_high + h_state,
                h_state,
            ),
            decimals=3,
        )
        self.n_states = self.state_space_h.shape[0]
        self.h_state = h_state

        # get initial state index 
        self.get_idx_state_init()

        # get target set indices
        self.get_idx_target_set()

    def discretize_action_space(self, h_action):

        # discretize action space
        self.action_space_h = np.arange(
            self.action_space_low,
            self.action_space_high + h_action,
            h_action,
        )
        self.n_actions = self.action_space_h.shape[0]
        self.h_action = h_action

        # get null action index
        self.get_idx_null_action()

    def get_state_idx(self, state):

        # array convertion
        state = np.asarray(state)

        # scalar input
        if state.ndim == 0:
            state = state[np.newaxis, np.newaxis]

        # array input
        elif state.ndim == 1:
            state = state[np.newaxis]

        idx = np.floor(
            (np.clip(
                state,
                self.state_space_low,
                self.state_space_high - 2 * self.h_state
            ) + self.state_space_high) / self.h_state).astype(int)
        idx = idx[:, 0]
        return idx

    def get_action_idx(self, action):
        # array convertion
        action = np.asarray(action)

        # scalar input
        if action.ndim == 0:
            action = action[np.newaxis, np.newaxis]

        # array input
        elif action.ndim == 1:
            action = action[np.newaxis]

        return np.argmin(np.abs(self.action_space_h - action), axis=1)

    def get_idx_state_init(self):
        self.idx_state_init = self.get_state_idx(self.state_init)

    def get_idx_target_set(self):
        #self.idx_a = np.where(self.state_space_h <= self.a_rb)[0]
        #self.idx_b = np.where(self.state_space_h >= self.b_lb)[0]
        #self.idx_ts = np.where((self.state_space_h <= self.a_rb) | (self.state_space_h >= self.b_lb))[0]
        #self.idx_not_ts = np.where((self.state_space_h > self.a_rb) & (self.state_space_h < self.b_lb))[0]

        # indices of domain_h in tha target set A
        self.idx_ts_a = np.where(
            (self.state_space_h >= self.target_set_a[0]) & (self.state_space_h <= self.target_set_a[1])
        )[0]

        # indices of domain_h in tha target set B
        self.idx_ts_b = np.where(
            (self.state_space_h >= self.target_set_b[0]) & (self.state_space_h <= self.target_set_b[1])
        )[0]

        # indices of the discretized domain corresponding to the target set
        self.idx_ts = np.where(
            ((self.state_space_h >= self.target_set_a[0]) & (self.state_space_h <= self.target_set_a[1])) |
            ((self.state_space_h >= self.target_set_b[0]) & (self.state_space_h <= self.target_set_b[1]))
        )[0]

        # indices of the discretized domain corresponding to the target set
        self.idx_not_ts = np.where(
            ((self.state_space_h < self.target_set_a[0]) | (self.state_space_h > self.target_set_a[1])) &
            ((self.state_space_h < self.target_set_b[0]) | (self.state_space_h > self.target_set_b[1]))
        )[0]

    def get_idx_null_action(self):
        self.idx_null_action = self.get_action_idx(np.zeros((1, 1)))

    def get_greedy_actions(self, q_table):

        # compute greedy action by following the q-table
        idx_actions = np.argmax(q_table, axis=1)
        greedy_actions = self.action_space_h[idx_actions]

        # set actions in the target set to 0
        greedy_actions[self.idx_ts] = 0.
        return greedy_actions

    def get_hjb_solver(self, h_hjb=0.001):
        from sde_hjb_solver.controlled_sde_1d import DoubleWellCommittor1D as SDE1D
        from sde_hjb_solver.hjb_solver_1d_st import SolverHJB1D

        # initialize controlled sde object
        sde = SDE1D(
            beta=self.beta,
            alpha=self.alpha,
            domain=(-2,  2),
            target_set_a=(-2, -1),
            target_set_b=(1, 2),
        )

        # initialize hjb solver object
        sol_hjb = SolverHJB1D(sde, h=1e-2)

        # load  hjb solver
        sol_hjb.load()


        # if hjb solver has different discretization step coarse solution
        if sol_hjb.sde.h < self.h_state:

            # discretization step ratio
            k = int(self.h_state / sol_hjb.sde.h)
            assert self.state_space_h.shape == sol_hjb.u_opt[::k].shape, ''

            sol_hjb.value_function = sol_hjb.value_function[::k]
            sol_hjb.u_opt = sol_hjb.u_opt[::k]

        return sol_hjb
