import os
import shutil
import sys

import numpy as np

from rl_sde_is.config import PROJECT_ROOT_DIR, DATA_ROOT_DIR

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

def save_data(dir_path, data_dict):
    file_path = os.path.join(dir_path, 'agent.npz')
    np.savez(file_path, **data_dict)

def load_data(dir_path):
    try:
        file_path = os.path.join(dir_path, 'agent.npz')
        data = dict(np.load(file_path, allow_pickle=True))
        for file_name in data.keys():
            if data[file_name].ndim == 0:
                data[file_name] = data[file_name].item()
        return data
    except FileNotFoundError as e:
        print(e)
        sys.exit()

def get_initial_point_str(env):
    if not env.is_state_init_sampled:
        initial_point_str = 'initial-state_{:2.1f}'.format(env.state_init.item())
    else:
        initial_point_str = 'explorable-starts'

    return initial_point_str

def get_alpha_str(**kwargs):
    if not kwargs['constant_alpha']:
        alpha_str = 'alpha_not-const'
    else:
        alpha_str = 'alpha_{:1.2f}'.format(kwargs['alpha'])
    return alpha_str

def get_eps_str(**kwargs):
    if kwargs['eps_type'] == 'constant':
        eps_str = os.path.join(
            'eps_const',
            'eps-init_{:0.1f}'.format(kwargs['eps_init']),
        )
    elif kwargs['eps_type'] == 'harmonic':
        eps_str = 'eps_harmonic'
    elif kwargs['eps_type'] == 'linear-decay':
        eps_str = os.path.join(
            'eps_linear-decay',
            'eps-min_{:0.1f}'.format(kwargs['eps_min']),
        )
    elif kwargs['eps_type'] == 'exp-decay':
        eps_str = os.path.join(
            'eps_exp-decay',
            'eps-init_{:0.1f}'.format(kwargs['eps_init']),
            'eps-decay_{:0.4f}'.format(kwargs['eps_decay']),
        )
    return eps_str


def get_random_dir_path(**kwargs):
    '''
    '''

    # set dir path
    dir_path = os.path.join(
        get_data_dir(),
        'random',
        get_initial_point_str(**kwargs),
        'K_{:.0e}'.format(kwargs['n_episodes']),
    )

    # create dir path if not exists
    make_dir_path(dir_path)

    return dir_path

def get_dynamic_programming_dir_path(env, **kwargs):
    '''
    '''
    # set dir path
    dir_path = os.path.join(
        get_data_dir(),
        '{}'.format(kwargs['agent']),
        'h-state_{:.0e}'.format(env.h_state),
        'h-action_{:.0e}'.format(env.h_action),
        'it_{:.0e}'.format(kwargs['n_iterations']),
    )

    # create dir path if not exists
    make_dir_path(dir_path)

    return dir_path

def get_sarsa_lambda_dir_path(env, **kwargs):
    '''
    '''
    dir_path = os.path.join(
        get_data_dir(),
        'sarsa-lambda',
        'h-state_{:.0e}'.format(env.h_state),
        'h-action_{:.0e}'.format(env.h_action),
        get_initial_point_str(env),
        'lr_{:1.2f}'.format(kwargs['lr']),
        'lambda_{:0.1f}'.format(kwargs['lam']),
        get_eps_str(**kwargs),
        'K_{:.0e}'.format(kwargs['n_episodes']),
    )

    # create dir path if not exists
    make_dir_path(dir_path)

    return dir_path

def get_qlearning_dir_path(env, **kwargs):
    '''
    '''

    dir_path = os.path.join(
        get_data_dir(),
        '{}'.format(kwargs['agent']),
        'h-state_{:.0e}'.format(env.h_state),
        'h-action_{:.0e}'.format(env.h_action),
        get_initial_point_str(env),
        'lr_{:1.2f}'.format(kwargs['lr']),
        get_eps_str(**kwargs),
        'K_{:.0e}'.format(kwargs['n_episodes']),
    )

    # create dir path if not exists
    make_dir_path(dir_path)

    return dir_path

def get_qlearning_batch_dir_path(env, **kwargs):
    '''
    '''

    dir_path = os.path.join(
        get_data_dir(),
        '{}'.format(kwargs['agent']),
        'h-state_{:.0e}'.format(env.h_state),
        'h-action_{:.0e}'.format(env.h_action),
        get_initial_point_str(env),
        'lr_{:1.2f}'.format(kwargs['lr']),
        get_eps_str(**kwargs),
        'epochs_{:.0e}'.format(kwargs['n_epochs']),
        'K_{:.0e}'.format(kwargs['batch_size']),
    )

    # create dir path if not exists
    make_dir_path(dir_path)

    return dir_path

def get_dqn_dir_path(env, **kwargs):
    '''
    '''

    dir_path = os.path.join(
        get_data_dir(),
        '{}'.format(kwargs['agent']),
        'h-action_{:.0e}'.format(env.h_action),
        get_initial_point_str(env),
        'lr_{:.1e}'.format(kwargs['lr']),
        'epochs_{:.0e}'.format(kwargs['n_epochs']),
        'K_{:.0e}'.format(kwargs['batch_size']),
    )

    # create dir path if not exists
    make_dir_path(dir_path)

    return dir_path

def get_reinforce_det_dir_path(env, **kwargs):
    '''
    '''

    dir_path = os.path.join(
        get_data_dir(),
        '{}'.format(kwargs['agent']),
        get_initial_point_str(env),
        'K_{:.0e}'.format(kwargs['batch_size']),
        'lr{:.1e}'.format(kwargs['lr']),
        'it_{:.0e}'.format(kwargs['n_iterations']),
        'seed_{:1d}'.format(kwargs['seed']),
    )

    # create dir path if not exists
    make_dir_path(dir_path)

    return dir_path

def get_ddpg_dir_path(env, **kwargs):
    '''
    '''

    if 'n_episodes' in kwargs.keys():
        iter_str = 'n-episodes_{:.0e}'.format(kwargs['n_episodes'])
    elif 'n_total_steps' in kwargs.keys():
        iter_str = 'n-total-steps_{:.0e}'.format(kwargs['n_total_steps'])
    else:
        iter_str = ''

    dir_path = os.path.join(
        get_data_dir(),
        '{}'.format(kwargs['agent']),
        get_initial_point_str(env),
        'K_{:.0e}'.format(kwargs['batch_size']),
        'lr-actor_{:.1e}'.format(kwargs['lr_actor']),
        'lr-critic_{:.1e}'.format(kwargs['lr_critic']),
        iter_str,
        'seed_{:1d}'.format(kwargs['seed']),
    )

    # create dir path if not exists
    make_dir_path(dir_path)

    return dir_path
