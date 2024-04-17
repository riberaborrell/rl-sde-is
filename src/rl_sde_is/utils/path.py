import os
import shutil
import sys

import torch
import numpy as np

from rl_sde_is.utils.config import PROJECT_ROOT_DIR, DATA_ROOT_DIR

def get_project_dir():
    ''' returns the absolute path of the repository's directory
    '''
    return PROJECT_ROOT_DIR

def get_data_dir():
    ''' returns the absolute path of the repository's data directory
    '''
    return DATA_ROOT_DIR


def make_dir_path(dir_path):
    ''' Create directories of the given path if they do not already exist
    '''
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

def empty_dir(dir_path):
    ''' Remove all files in the directory from the given path
    '''
    if os.path.isdir(dir_path):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete {}. Reason: {}'.format((file_path, e)))

def get_fig_notebooks_dir_path():
    ''' returns the absolute path of the notebooks directory figures
    '''

    # get dir path
    dir_path = os.path.join(get_data_dir(), 'notebooks')

    # create dir path if not exists
    make_dir_path(dir_path)

    return dir_path

def save_data(data_dict, rel_dir_path):
    file_path = os.path.join(get_data_dir(), rel_dir_path, 'agent.npz')
    np.savez(file_path, **data_dict)

def load_data(rel_dir_path):
    try:
        file_path = os.path.join(get_data_dir(), rel_dir_path, 'agent.npz')
        data = dict(np.load(file_path, allow_pickle=True))
        for file_name in data.keys():
            if data[file_name].ndim == 0:
                data[file_name] = data[file_name].item()
        data['rel_dir_path'] = rel_dir_path
        return data
    except FileNotFoundError as e:
        print(e)
        sys.exit()

def save_model(model, rel_dir_path, file_name):
    torch.save(
        model.state_dict(),
        os.path.join(get_data_dir(), rel_dir_path, file_name),
    )

def load_model(model, rel_dir_path, file_name):
    model.load_state_dict(torch.load(os.path.join(get_data_dir(), rel_dir_path, file_name)))


def get_rel_dir_path(env_str, algorithm_name, param_str):

    # relative directory path
    rel_dir_path = os.path.join(
        env_str,
        algorithm_name,
        param_str,
    )

    # create dir path if not exists
    make_dir_path(os.path.join(get_data_dir(), rel_dir_path))

    return rel_dir_path

def get_initial_point_str(env):
    if env.state_init_dist == 'delta':
        string = 'init-state{:2.1f}_'.format(env.state_init[0, 0].item())
    elif env.state_init_dist == 'uniform':
        string = 'uniform_'

    return string

def get_baseline_str(env):
    if env.reward_type == 'baseline':
        string = 'baseline_'
    else:
        string = ''

    return string

def get_lr_str(**kwargs):
    if not kwargs['constant_lr']:
        string = 'lr-not-const_'
    else:
        string = 'lr{:1.2f}_'.format(kwargs['lr'])
    return string

def get_eps_str(**kwargs):
    if kwargs['eps_type'] == 'constant':
        string = 'eps-const_' \
                + 'eps-init{:0.1f}_'.format(kwargs['eps_init'])
    elif kwargs['eps_type'] == 'harmonic':
        string = 'eps_harmonic_'
    elif kwargs['eps_type'] == 'linear-decay':
        string = 'eps-linear-decay_' \
                + 'eps-min{:0.1f}_'.format(kwargs['eps_min'])
    elif kwargs['eps_type'] == 'exp-decay':
        string = 'eps-exp-decay_' \
                + 'eps-init{:0.1f}_'.format(kwargs['eps_init']) \
                + 'eps-decay{:0.4f}_'.format(kwargs['eps_decay'])
    return string

def get_iter_str(**kwargs):
    if 'n_episodes' in kwargs.keys():
        string = 'n-episodes{:.0e}_'.format(kwargs['n_episodes'])
    elif 'n_total_steps' in kwargs.keys():
        string = 'n-total-steps{:.0e}_'.format(kwargs['n_total_steps'])
    elif 'n_iterations' in kwargs.keys():
        string = 'n-iter{:.0e}_'.format(kwargs['n_iterations'])
    else:
        string = ''
    return string

def get_seed_str(**kwargs):
    if 'seed' not in kwargs.keys() or not kwargs['seed']:
        string = 'seedNone'.format(kwargs['seed'])
    else:
        string = 'seed{:1d}'.format(kwargs['seed'])
    return string

def get_agent_dir_path(env, **kwargs):
    '''
    '''

    # set parameters string
    param_str = get_initial_point_str(env) \
              + 'dt{:.0e}_'.format(env.dt) \
              + 'K{:.0e}'.format(kwargs['batch_size']) \
              + get_seed_str(**kwargs)

    return get_rel_dir_path(env, kwargs['agent'], param_str)

def get_dynamic_programming_tables_dir_path(env):
    '''
    '''
    # set parameters string
    param_str = 'h-state{:.0e}_'.format(env.h_state) \
              + 'h-action{:.0e}_'.format(env.h_action) \
              + 'dt{:.0e}'.format(env.dt)

    return get_rel_dir_path(env, 'dp-tables', param_str)

def get_dynamic_programming_dir_path(env, **kwargs):
    '''
    '''
    # set parameters string
    param_str = 'h-state{:.0e}_'.format(env.h_state) \
              + 'h-action{:.0e}_'.format(env.h_action) \
              + 'dt{:.0e}_'.format(env.dt) \
              + 'n-it{:.0e}'.format(kwargs['n_iterations'])

    return get_rel_dir_path(env, kwargs['agent'], param_str)

def get_tabular_td_prediction_dir_path(env, **kwargs):
    '''
    '''

    # set parameters string
    param_str = 'h-state{:.0e}_'.format(env.h_state) \
              + 'h-action{:.0e}_'.format(env.h_action) \
              + 'dt{:.0e}_'.format(env.dt) \
              + get_initial_point_str(env) \
              + 'lr{:1.2f}_'.format(kwargs['lr']) \
              + 'K{:.0e}'.format(kwargs['n_episodes']) \
              + get_seed_str(**kwargs)

    return get_rel_dir_path(env, kwargs['agent'], param_str)

def get_semi_gradient_td_prediction_dir_path(env, **kwargs):
    '''
    '''

    # set parameters string
    param_str = 'dt{:.0e}_'.format(env.dt) \
              + get_initial_point_str(env) \
              + 'lr{:1.2f}_'.format(kwargs['lr']) \
              + 'K{:.0e}'.format(kwargs['n_episodes']) \
              + get_seed_str(**kwargs)

    return get_rel_dir_path(env, kwargs['agent'], param_str)

def get_tabular_mc_prediction_dir_path(env, **kwargs):
    '''
    '''

    # set parameters string
    param_str = 'h-state{:.0e}_'.format(env.h_state) \
              + 'h-action{:.0e}_'.format(env.h_action) \
              + 'dt{:.0e}_'.format(env.dt) \
              + get_initial_point_str(env) \
              + 'K{:.0e}'.format(kwargs['n_episodes']) \
              + get_seed_str(**kwargs)

    return get_rel_dir_path(env, kwargs['agent'], param_str)


def get_sarsa_lambda_dir_path(env, **kwargs):
    '''
    '''
    # set parameters string
    param_str = 'h-state{:.0e}_'.format(env.h_state) \
              + 'h-action{:.0e}_'.format(env.h_action) \
              + get_initial_point_str(env) \
              + get_baseline_str(env) \
              + 'lr{:1.2f}_'.format(kwargs['lr']) \
              + 'lambda{:0.1f}_'.format(kwargs['lam']) \
              + get_eps_str(**kwargs) \
              + 'K{:.0e}'.format(kwargs['n_episodes']) \
              + get_seed_str(**kwargs)

    return get_rel_dir_path(env, kwargs['agent'], param_str)

def get_mc_learning_dir_path(env, **kwargs):
    '''
    '''
    # set parameters string
    param_str = 'h-state{:.0e}_'.format(env.h_state) \
              + 'h-action{:.0e}_'.format(env.h_action) \
              + get_initial_point_str(env) \
              + 'lr{:1.2f}_'.format(kwargs['lr']) \
              + get_eps_str(**kwargs) \
              + 'K{:.0e}'.format(kwargs['n_episodes']) \
              + get_seed_str(**kwargs)

    return get_rel_dir_path(env, kwargs['agent'], param_str)

def get_qlearning_dir_path(env, **kwargs):
    '''
    '''

    # set parameters string
    param_str = 'h-state{:.0e}_'.format(env.h_state) \
              + 'h-action{:.0e}_'.format(env.h_action) \
              + get_initial_point_str(env) \
              + get_baseline_str(env) \
              + 'lr{:1.2f}_'.format(kwargs['lr']) \
              + get_eps_str(**kwargs) \
              + 'K{:.0e}'.format(kwargs['n_episodes']) \
              + get_seed_str(**kwargs)

    return get_rel_dir_path(env, kwargs['agent'], param_str)

#TODO: check
def get_qlearning_batch_dir_path(env, **kwargs):
    '''
    '''

    # set parameters string
    param_str = 'h-state{:.0e}_'.format(env.h_state) \
              + 'h-action{:.0e}_'.format(env.h_action) \
              + get_initial_point_str(env) \
              + 'lr{:1.2f}_'.format(kwargs['lr']) \
              + get_eps_str(**kwargs) \
              + 'epochs{:.0e}_'.format(kwargs['n_epochs']) \
              + 'K{:.0e}'.format(kwargs['n_episodes'])

    return get_rel_dir_path(env, kwargs['agent'], param_str)

#TODO: revise
def get_dqn_dir_path(env, **kwargs):
    '''
    '''

    # set parameters string
    param_str = 'h-action{:.0e}_'.format(env.h_action) \
              + 'dt{:.0e}_'.format(env.dt) \
              + get_initial_point_str(env) \
              + 'lr{:.1e}_'.format(kwargs['lr']) \
              + get_iter_str(**kwargs) \
              + 'K{:.0e}'.format(kwargs['batch_size']) \

    return get_rel_dir_path(env, kwargs['agent'], param_str)

def get_reinforce_stoch_dir_path(env, **kwargs):
    '''
    '''
    if kwargs['agent'] == 'reinforce-stochastic-discrete':
        h_action_str = 'h-action{:.0e}_'.format(env.h_action)
    else:
        h_action_str = ''

    # set parameters string
    param_str = h_action_str \
              + 'dt{:.0e}_'.format(env.dt) \
              + get_initial_point_str(env) \
              + 'lr{:.1e}_'.format(kwargs['lr']) \
              + 'K{:.0e}_'.format(kwargs['batch_size']) \
              + get_iter_str(**kwargs) \
              + get_seed_str(**kwargs)

    return get_rel_dir_path(env, kwargs['agent'], param_str)


def get_reinforce_det_dir_path(env, **kwargs):
    '''
    '''
    # set parameters string
    param_str = get_initial_point_str(env) \
              + get_baseline_str(env) \
              + 'gamma{:.3f}_'.format(kwargs['gamma']) \
              + 'hidden-size{:d}_'.format(kwargs['d_hidden_layer']) \
              + 'K{:.0e}_'.format(kwargs['batch_size']) \
              + 'lr{:.1e}_'.format(kwargs['lr']) \
              + get_iter_str(**kwargs) \
              + get_seed_str(**kwargs)

    return get_rel_dir_path(env, kwargs['agent'], param_str)

def get_dpg_dir_path(env, **kwargs):
    '''
    '''

    # set parameters string
    param_str = get_initial_point_str(env) \
              + 'gamma{:.3f}_'.format(kwargs['gamma']) \
              + 'hidden-size{:d}_'.format(kwargs['d_hidden_layer']) \
              + 'K{:.0e}_'.format(kwargs['batch_size']) \
              + 'lr-actor{:.1e}_'.format(kwargs['lr_actor']) \
              + get_iter_str(**kwargs) \
              + get_seed_str(**kwargs)

    return get_rel_dir_path(env, kwargs['agent'], param_str)

def get_ddpg_dir_path(env, **kwargs):
    '''
    '''

    # set parameters string
    param_str = get_initial_point_str(env) \
              + 'gamma{:.3f}_'.format(kwargs['gamma']) \
              + 'hidden-size{:d}_'.format(kwargs['d_hidden_layer']) \
              + 'noise-scale{:.1e}_'.format(kwargs['noise_scale']) \
              + 'K{:.0e}_'.format(kwargs['batch_size']) \
              + 'lr-actor{:.1e}_'.format(kwargs['lr_actor']) \
              + 'lr-critic{:.1e}_'.format(kwargs['lr_critic']) \
              + get_iter_str(**kwargs) \
              + get_seed_str(**kwargs)

    return get_rel_dir_path(env, kwargs['agent'], param_str)

def get_td3_dir_path(env, **kwargs):
    '''
    '''

    # set parameters string
    param_str = get_initial_point_str(env) \
              + get_baseline_str(env) \
              + 'gamma{:.3f}_'.format(kwargs['gamma']) \
              + 'hidden-size{:d}_'.format(kwargs['d_hidden_layer']) \
              + 'n-steps-lim{:.1e}_'.format(kwargs['n_steps_lim']) \
              + 'expl-noise{:.1f}_'.format(kwargs['expl_noise_init']) \
              + 'policy-delay{:d}_'.format(kwargs['policy_delay']) \
              + 'target-noise{:.1f}_'.format(kwargs['target_noise']) \
              + 'polyak{:.3f}_'.format(kwargs['polyak']) \
              + 'K{:.0e}_'.format(kwargs['batch_size']) \
              + 'lr-actor{:.1e}_'.format(kwargs['lr_actor']) \
              + 'lr-critic{:.1e}_'.format(kwargs['lr_critic']) \
              + get_iter_str(**kwargs) \
              + get_seed_str(**kwargs)

    return get_rel_dir_path(env.name, kwargs['agent'], param_str)