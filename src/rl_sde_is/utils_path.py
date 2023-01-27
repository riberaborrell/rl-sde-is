import os
import shutil
import sys

import torch
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


def get_initial_point_str(env):
    if not env.is_state_init_sampled:
        initial_point_str = 'init-state{:2.1f}_'.format(env.state_init[0, 0].item())
    else:
        initial_point_str = 'explorable-starts_'

    return initial_point_str

def get_lr_str(**kwargs):
    if not kwargs['constant_lr']:
        lr_str = 'lr-not-const_'
    else:
        lr_str = 'lr{:1.2f}_'.format(kwargs['lr'])
    return lr_str

def get_eps_str(**kwargs):
    if kwargs['eps_type'] == 'constant':
        eps_str = 'eps-const_' \
                + 'eps-init{:0.1f}_'.format(kwargs['eps_init'])
    elif kwargs['eps_type'] == 'harmonic':
        eps_str = 'eps_harmonic_'
    elif kwargs['eps_type'] == 'linear-decay':
        eps_str = 'eps-linear-decay_' \
                + 'eps-min{:0.1f}_'.format(kwargs['eps_min'])
    elif kwargs['eps_type'] == 'exp-decay':
        eps_str = 'eps-exp-decay_' \
                + 'eps-init{:0.1f}_'.format(kwargs['eps_init']) \
                + 'eps-decay{:0.4f}_'.format(kwargs['eps_decay'])
    return eps_str

def get_iter_str(**kwargs):
    if 'n_episodes' in kwargs.keys():
        iter_str = 'n-episodes{:.0e}_'.format(kwargs['n_episodes'])
    elif 'n_total_steps' in kwargs.keys():
        iter_str = 'n-total-steps{:.0e}_'.format(kwargs['n_total_steps'])
    elif 'n_iterations' in kwargs.keys():
        iter_str = 'n-iter{:.0e}_'.format(kwargs['n_iterations'])
    else:
        iter_str = ''
    return iter_str

def get_agent_dir_path(env, **kwargs):
    '''
    '''

    # set parameters string
    param_str = get_initial_point_str(env) \
              + 'dt{:.0e}_'.format(env.dt) \
              + 'n-episodes{:.0e}_'.format(kwargs['n_episodes']) \
              + 'seed{:1d}'.format(kwargs['seed'])


    # set dir path
    rel_dir_path = os.path.join(
        env.name,
        'agent-{}'.format(kwargs['agent']),
        param_str,
    )

    # create dir path if not exists
    make_dir_path(os.path.join(get_data_dir(), rel_dir_path))

    return rel_dir_path

def get_dynamic_programming_dir_path(env, **kwargs):
    '''
    '''
    # set parameters string
    param_str = 'h-state{:.0e}_'.format(env.h_state) \
              + 'h-action{:.0e}_'.format(env.h_action) \
              + 'dt{:.0e}_'.format(env.dt) \
              + 'n-it{:.0e}'.format(kwargs['n_iterations'])


    # set dir path
    rel_dir_path = os.path.join(
        env.name,
        kwargs['agent'],
        param_str,
    )

    # create dir path if not exists
    make_dir_path(os.path.join(get_data_dir(), rel_dir_path))

    return rel_dir_path

def get_sarsa_lambda_dir_path(env, **kwargs):
    '''
    '''
    # set parameters string
    param_str = 'h-state{:.0e}_'.format(env.h_state) \
              + 'h-action{:.0e}_'.format(env.h_action) \
              + get_initial_point_str(env) \
              + 'lr{:1.2f}_'.format(kwargs['lr']) \
              + 'lambda{:0.1f}_'.format(kwargs['lam']) \
              + get_eps_str(**kwargs) \
              + 'K{:.0e}_'.format(kwargs['n_episodes'])

    dir_path = os.path.join(
        get_data_dir(),
        env.name,
        'sarsa-lambda',
        param_str,
    )

    # create dir path if not exists
    make_dir_path(dir_path)

    return dir_path

def get_qlearning_dir_path(env, **kwargs):
    '''
    '''

    # set parameters string
    param_str = 'h-state{:.0e}_'.format(env.h_state) \
              + 'h-action{:.0e}_'.format(env.h_action) \
              + get_initial_point_str(env) \
              + 'lr{:1.2f}_'.format(kwargs['lr']) \
              + get_eps_str(**kwargs) \
              + 'K{:.0e}_'.format(kwargs['n_episodes'])


    dir_path = os.path.join(
        get_data_dir(),
        env.name,
        kwargs['agent'],
        param_str,
    )

    # create dir path if not exists
    make_dir_path(dir_path)

    return dir_path

def get_qlearning_batch_dir_path(env, **kwargs):
    '''
    '''

    dir_path = os.path.join(
        get_data_dir(),
        env.name,
        kwargs['agent'],
        'h-state{:.0e}_'.format(env.h_state),
        'h-action{:.0e}_'.format(env.h_action),
        get_initial_point_str(env),
        'lr{:1.2f}_'.format(kwargs['lr']),
        get_eps_str(**kwargs),
        'epochs{:.0e}_'.format(kwargs['n_epochs']),
        'K{:.0e}_'.format(kwargs['batch_size']),
    )

    # create dir path if not exists
    make_dir_path(dir_path)

    return dir_path

def get_dqn_dir_path(env, **kwargs):
    '''
    '''

    dir_path = os.path.join(
        get_data_dir(),
        env.name,
        kwargs['agent'],
        'h-action{:.0e}_'.format(env.h_action),
        get_initial_point_str(env),
        'lr{:.1e}_'.format(kwargs['lr']),
        'epochs{:.0e}_'.format(kwargs['n_epochs']),
        'K{:.0e}_'.format(kwargs['batch_size']),
    )

    # create dir path if not exists
    make_dir_path(dir_path)

    return dir_path

def get_reinforce_stoch_dir_path(env, **kwargs):
    '''
    '''
    if kwargs['agent'] == 'reinforce-stochastic-discrete':
        h_action_str = 'h-action{:.0e}_'.format(env.h_action)
    else:
        h_action_str = ''

    # set parameters string
    param_str = get_initial_point_str(env) \
              + h_action_str \
              + 'K{:.0e}_'.format(kwargs['batch_size']) \
              + 'lr{:.1e}_'.format(kwargs['lr']) \
              + get_iter_str(**kwargs) \
              + 'seed{:1d}'.format(kwargs['seed'])

    dir_path = os.path.join(
        get_data_dir(),
        env.name,
        kwargs['agent'],
        param_str,
    )

    # create dir path if not exists
    make_dir_path(dir_path)

    return dir_path

def get_reinforce_det_dir_path(env, **kwargs):
    '''
    '''
    # set parameters string
    param_str = get_initial_point_str(env) \
              + 'gamma{:.3f}_'.format(kwargs['gamma']) \
              + 'hidden-size{:d}_'.format(kwargs['d_hidden_layer']) \
              + 'K{:.0e}_'.format(kwargs['batch_size']) \
              + 'lr{:.1e}_'.format(kwargs['lr']) \
              + get_iter_str(**kwargs) \
              + 'seed{:1d}'.format(kwargs['seed'])

    rel_dir_path = os.path.join(
        env.name,
        kwargs['agent'],
        param_str,
    )

    # create dir path if not exists
    make_dir_path(os.path.join(get_data_dir(), rel_dir_path))

    return rel_dir_path

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
              + 'seed{:1d}'.format(kwargs['seed'])

    rel_dir_path = os.path.join(
        env.name,
        kwargs['agent'],
        param_str,
    )

    # create dir path if not exists
    make_dir_path(os.path.join(get_data_dir(), rel_dir_path))

    return rel_dir_path

def get_td3_dir_path(env, **kwargs):
    '''
    '''

    # set parameters string
    param_str = get_initial_point_str(env) \
              + 'gamma{:.3f}_'.format(kwargs['gamma']) \
              + 'hidden-size{:d}_'.format(kwargs['d_hidden_layer']) \
              + 'expl-noise{:.1f}_'.format(kwargs['expl_noise_init']) \
              + 'policy-delay{:d}_'.format(kwargs['policy_delay']) \
              + 'target-noise{:.1f}_'.format(kwargs['target_noise']) \
              + 'polyak{:.3f}_'.format(kwargs['polyak']) \
              + 'K{:.0e}_'.format(kwargs['batch_size']) \
              + 'lr-actor{:.1e}_'.format(kwargs['lr_actor']) \
              + 'lr-critic{:.1e}_'.format(kwargs['lr_critic']) \
              + get_iter_str(**kwargs) \
              + 'seed{:1d}'.format(kwargs['seed'])

    rel_dir_path = os.path.join(
        env.name,
        kwargs['agent'],
        param_str,
    )

    # create dir path if not exists
    make_dir_path(os.path.join(get_data_dir(), rel_dir_path))

    return rel_dir_path
