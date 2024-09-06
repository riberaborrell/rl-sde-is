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

def save_data(data_dict, dir_path, file_name: str = 'agent.npz'):
    file_path = os.path.join(get_data_dir(), dir_path, file_name)
    np.savez(file_path, **data_dict)

def load_data(dir_path, file_name: str = 'agent.npz'):
    try:
        file_path = os.path.join(get_data_dir(), dir_path, file_name)
        data = dict(np.load(file_path, allow_pickle=True))
        for file_name in data.keys():
            if data[file_name].ndim == 0:
                data[file_name] = data[file_name].item()
        data['dir_path'] = dir_path
        return data
    except FileNotFoundError as e:
        print(e)
        sys.exit()

def save_model(model, dir_path, file_name):
    torch.save(
        model.state_dict(),
        os.path.join(get_data_dir(), dir_path, file_name),
    )

def load_model(model, dir_path, file_name):
    model.load_state_dict(torch.load(os.path.join(get_data_dir(), dir_path, file_name)))


def get_dir_path(env_str, algorithm_name, param_str):

    # directory path
    dir_path = os.path.join(
        env_str,
        algorithm_name,
        param_str,
    )

    # create dir path if not exists
    make_dir_path(os.path.join(get_data_dir(), dir_path))

    return dir_path

def get_baseline_str(env, **kwargs):
    if env.reward_type == 'baseline':
        string = 'baseline-factor{}_'.format(kwargs['baseline_scale_factor'])
    else:
        string = ''
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

def get_model_arch_str(**kwargs):
    string = ''
    if 'n_layers' in kwargs.keys():
        string += 'n-layers{:d}_'.format(kwargs['n_layers'])
    if 'd_hidden_layer' in kwargs.keys():
        string += 'hidden-size{:d}_'.format(kwargs['d_hidden_layer'])
    if 'theta_init' in kwargs.keys():
        string += 'theta-init-{}_'.format(kwargs['theta_init'])
    return string

def get_action_limit_str(**kwargs):
    if 'action_limit' in kwargs.keys() and kwargs['action_limit'] is not None:
        string = 'action-limit{:.1f}_'.format(kwargs['action_limit'])
    else:
        string = ''
    return string

def get_lr_and_batch_size_str(**kwargs):
    string = ''
    string += 'lr{:.1e}_'.format(kwargs['lr']) if 'lr' in kwargs.keys() else ''
    string += 'lr-init{:.1e}_'.format(kwargs['lr_init']) if 'lr_init' in kwargs.keys() else ''
    string += 'lr-actor{:.1e}_'.format(kwargs['lr_actor']) if 'lr_actor' in kwargs.keys() else ''
    string += 'lr-critic{:.1e}_'.format(kwargs['lr_critic']) if 'lr_critic' in kwargs.keys() else ''
    string += 'K{:d}_'.format(int(kwargs['batch_size'])) if 'batch_size' in kwargs.keys() else ''
    string += 'mini-K{:d}_'.format(int(kwargs['mini_batch_size'])) \
               if 'mini_batch_size' in kwargs.keys() and kwargs['mini_batch_size'] is not None \
               else ''
    return string

def get_z_estimation_str(**kwargs):
    if 'on-policy' in kwargs['agent']:
        return 'z-estimated_' if kwargs['estimate_z'] else 'z-neglected_'
    else:
        return ''

def get_iter_str(**kwargs):
    if 'n_episodes' in kwargs.keys():
        string = 'n-episodes{:.0e}_'.format(kwargs['n_episodes'])
    elif 'n_total_steps' in kwargs.keys():
        string = 'n-total-steps{:.0e}_'.format(kwargs['n_total_steps'])
    elif 'n_grad_iterations' in kwargs.keys():
        string = 'n-grad-iter{:.0e}_'.format(kwargs['n_grad_iterations'])
    else:
        string = ''
    return string

def get_seed_str(**kwargs):
    if 'seed' not in kwargs.keys() or not kwargs['seed']:
        string = 'seedNone'.format(kwargs['seed'])
    else:
        string = 'seed{:1d}'.format(kwargs['seed'])
    return string

def get_replay_memory_str(**kwargs):
    string = ''
    string += 'replay-size{:.1e}_'.format(kwargs['replay_size']) if 'replay_size' in kwargs.keys() else ''
    string += 'learning-starts{:d}_'.format(kwargs['learning_starts']) if 'learning_starts' in kwargs.keys() else ''
    return string

def get_agent_dir_path(env, **kwargs):
    '''
    '''

    # set parameters string
    param_str = 'dt{:.0e}_'.format(env.dt) \
              + 'K{:.0e}'.format(kwargs['batch_size']) \
              + get_seed_str(**kwargs)

    return get_dir_path(env, kwargs['agent'], param_str)

def get_dynamic_programming_tables_dir_path(env):
    '''
    '''
    # set parameters string
    param_str = 'h-state{:.0e}_'.format(env.h_state) \
              + 'h-action{:.0e}_'.format(env.h_action) \
              + 'dt{:.0e}'.format(env.dt)

    return get_dir_path(env.unwrapped.__str__(), 'dp-tables', param_str)

def get_dynamic_programming_dir_path(env, **kwargs):
    '''
    '''
    # set parameters string
    param_str = 'h-state{:.0e}_'.format(env.h_state) \
              + 'h-action{:.0e}_'.format(env.h_action) \
              + 'dt{:.0e}_'.format(env.dt) \
              + 'n-it{:.0e}'.format(kwargs['n_iterations'])

    return get_dir_path(env.unwrapped.__str__(), kwargs['agent'], param_str)

def get_tabular_td_prediction_dir_path(env, **kwargs):
    '''
    '''

    # set parameters string
    param_str = 'h-state{:.0e}_'.format(env.h_state) \
              + 'h-action{:.0e}_'.format(env.h_action) \
              + 'dt{:.0e}_'.format(env.dt) \
              + get_lr_and_batch_size_str(**kwargs) \
              + 'K{:.0e}'.format(kwargs['n_episodes']) \
              + get_seed_str(**kwargs)

    return get_dir_path(env.unwrapped.__str__(), kwargs['agent'], param_str)

def get_semi_gradient_td_prediction_dir_path(env, **kwargs):
    '''
    '''

    # set parameters string
    param_str = 'dt{:.0e}_'.format(env.dt) \
              + get_lr_and_batch_size_str(**kwargs) \
              + 'K{:.0e}'.format(kwargs['n_episodes']) \
              + get_seed_str(**kwargs)

    return get_dir_path(env.unwrapped.__str__(), kwargs['agent'], param_str)

def get_tabular_mc_prediction_dir_path(env, **kwargs):
    '''
    '''

    # set parameters string
    param_str = 'h-state{:.0e}_'.format(env.h_state) \
              + 'h-action{:.0e}_'.format(env.h_action) \
              + 'dt{:.0e}_'.format(env.dt) \
              + 'K{:.0e}'.format(kwargs['n_episodes']) \
              + get_seed_str(**kwargs)

    return get_dir_path(env.unwrapped.__str__(), kwargs['agent'], param_str)


def get_sarsa_lambda_dir_path(env, **kwargs):
    '''
    '''
    # set parameters string
    param_str = 'h-state{:.0e}_'.format(env.h_state) \
              + 'h-action{:.0e}_'.format(env.h_action) \
              + 'dt{:.0e}_'.format(env.dt) \
              + get_lr_and_batch_size_str(**kwargs) \
              + 'lambda{:0.1f}_'.format(kwargs['lam']) \
              + get_eps_str(**kwargs) \
              + 'K{:.0e}'.format(kwargs['n_episodes']) \
              + get_seed_str(**kwargs)

    return get_dir_path(env.unwrapped.__str__(), kwargs['agent'], param_str)

def get_mc_learning_dir_path(env, **kwargs):
    '''
    '''
    # set parameters string
    param_str = 'h-state{:.0e}_'.format(env.h_state) \
              + 'h-action{:.0e}_'.format(env.h_action) \
              + 'dt{:.0e}_'.format(env.dt) \
              + get_lr_and_batch_size_str(**kwargs) \
              + get_eps_str(**kwargs) \
              + 'K{:.0e}'.format(kwargs['n_episodes']) \
              + get_seed_str(**kwargs)

    return get_dir_path(env.unwrapped.__str__(), kwargs['agent'], param_str)

def get_qlearning_dir_path(env, **kwargs):
    '''
    '''

    # set parameters string
    param_str = 'h-state{:.0e}_'.format(env.h_state) \
              + 'h-action{:.0e}_'.format(env.h_action) \
              + 'dt{:.0e}_'.format(env.dt) \
              + get_lr_and_batch_size_str(**kwargs) \
              + get_eps_str(**kwargs) \
              + 'K{:.0e}'.format(kwargs['n_episodes']) \
              + get_seed_str(**kwargs)

    return get_dir_path(env.unwrapped.__str__(), kwargs['agent'], param_str)

#TODO: check
def get_qlearning_batch_dir_path(env, **kwargs):
    '''
    '''

    # set parameters string
    param_str = 'h-state{:.0e}_'.format(env.h_state) \
              + 'h-action{:.0e}_'.format(env.h_action) \
              + 'dt{:.0e}_'.format(env.dt) \
              + get_lr_and_batch_size_str(**kwargs) \
              + get_eps_str(**kwargs) \
              + 'epochs{:.0e}_'.format(kwargs['n_epochs']) \
              + 'K{:.0e}'.format(kwargs['n_episodes'])

    return get_dir_path(env.unwrapped.__str__(), kwargs['agent'], param_str)

def get_dqn_dir_path(env, **kwargs):
    '''
    '''

    # set parameters string
    param_str = 'h-action{:.0e}_'.format(env.h_action) \
              + 'dt{:.0e}_'.format(env.dt) \
              + 'gamma{:.3f}_'.format(kwargs['gamma']) \
              + get_model_arch_str(**kwargs) \
              + 'polyak{:.3f}_'.format(kwargs['polyak']) \
              + get_lr_and_batch_size_str(**kwargs) \
              + get_iter_str(**kwargs) \
              + get_seed_str(**kwargs)

    return get_dir_path(env.unwrapped.__str__(), kwargs['agent'], param_str)

def get_reinforce_discrete_dir_path(env, **kwargs):
    '''
    '''

    # set parameters string
    param_str = 'h-action{:.0e}_'.format(kwargs['h_action']) \
              + 'dt{:.0e}_'.format(env.dt) \
              + 'gamma{:.3f}_'.format(kwargs['gamma']) \
              + get_lr_and_batch_size_str(**kwargs) \
              + get_iter_str(**kwargs) \
              + get_seed_str(**kwargs)

    return get_dir_path(env.unwrapped.__str__(), kwargs['agent'], param_str)

def get_reinforce_stoch_dir_path(env, **kwargs):
    '''
    '''

    # set parameters string
    param_str = 'dt{:.0e}_'.format(env.dt) \
              + 'gamma{:.3f}_'.format(kwargs['gamma']) \
              + get_model_arch_str(**kwargs) \
              + 'policy-{}_'.format(kwargs['policy_type']) \
              + 'policy-noise{:.2f}_'.format(kwargs['policy_noise']) \
              + '{}_'.format(kwargs['return_type']) \
              + get_z_estimation_str(**kwargs) \
              + get_lr_and_batch_size_str(**kwargs) \
              + get_iter_str(**kwargs) \
              + 'learn-value{}_'.format(kwargs['learn_value']) \
              + get_seed_str(**kwargs)

    return get_dir_path(env.unwrapped.__str__(), kwargs['agent'], param_str)


def get_reinforce_det_dir_path(env, **kwargs):
    '''
    '''
    # set parameters string
    param_str = 'dt{:.0e}_'.format(env.dt) \
              + 'gamma{:.3f}_'.format(kwargs['gamma']) \
              + get_model_arch_str(**kwargs) \
              + '{}_'.format(kwargs['return_type']) \
              + get_z_estimation_str(**kwargs) \
              + get_lr_and_batch_size_str(**kwargs) \
              + get_iter_str(**kwargs) \
              + 'learn-value{}_'.format(kwargs['learn_value']) \
              + get_seed_str(**kwargs)

    return get_dir_path(env.unwrapped.__str__(), kwargs['agent'], param_str)

def get_model_based_dpg_dir_path(env, **kwargs):
    '''
    '''
    # set parameters string
    param_str = 'dt{:.0e}_'.format(env.dt) \
              + 'gamma{:.3f}_'.format(kwargs['gamma']) \
              + get_model_arch_str(**kwargs) \
              + '{}_'.format(kwargs['return_type']) \
              + 'z-estimated_' if kwargs['estimate_z'] else 'z-neglected_' \
              + 'n-steps-lim{:.1e}_'.format(kwargs['n_steps_lim']) \
              + get_lr_and_batch_size_str(**kwargs) \
              + get_iter_str(**kwargs) \
              + 'learn-value{}_'.format(kwargs['learn_value']) \
              + get_seed_str(**kwargs)

    return get_dir_path(env.unwrapped.__str__(), kwargs['agent'], param_str)

def get_dpg_dir_path(env, **kwargs):
    '''
    '''

    # set parameters string
    param_str = 'dt{:.0e}_'.format(env.dt) \
              + 'gamma{:.3f}_'.format(kwargs['gamma']) \
              + get_model_arch_str(**kwargs) \
              + get_lr_and_batch_size_str(**kwargs) \
              + get_iter_str(**kwargs) \
              + get_seed_str(**kwargs)

    return get_dir_path(env.name, kwargs['agent'], param_str)

def get_dpg_optimal_dir_path(env, **kwargs):
    '''
    '''

    # set parameters string
    param_str = 'dt{:.0e}_'.format(env.dt) \
              + 'gamma{:.3f}_'.format(kwargs['gamma']) \
              + get_model_arch_str(**kwargs) \
              + get_z_estimation_str(**kwargs) \
              + get_lr_and_batch_size_str(**kwargs) \
              + get_iter_str(**kwargs) \
              + get_seed_str(**kwargs)

    return get_dir_path(env.name, kwargs['agent'], param_str)

def get_ddpg_dir_path(env, **kwargs):
    '''
    '''

    # set parameters string
    param_str = 'dt{:.0e}_'.format(env.dt) \
              + 'gamma{:.3f}_'.format(kwargs['gamma']) \
              + get_model_arch_str(**kwargs) \
              + get_action_limit_str(**kwargs) \
              + 'expl-noise{:.1f}_'.format(kwargs['expl_noise']) \
              + 'polyak{:.3f}_'.format(kwargs['polyak']) \
              + get_lr_and_batch_size_str(**kwargs) \
              + get_iter_str(**kwargs) \
              + get_seed_str(**kwargs)

    return get_dir_path(env.unwrapped.__str__(), kwargs['agent'], param_str)

def get_td3_dir_path(env, **kwargs):
    '''
    '''

    # set parameters string
    param_str = 'dt{:.0e}_'.format(env.dt) \
              + 'gamma{:.3f}_'.format(kwargs['gamma']) \
              + get_model_arch_str(**kwargs) \
              + 'n-steps-lim{:.1e}_'.format(kwargs['n_steps_lim']) \
              + get_action_limit_str(**kwargs) \
              + 'expl-noise{:.1f}_'.format(kwargs['expl_noise_init']) \
              + 'policy-freq{:d}_'.format(kwargs['policy_freq']) \
              + 'target-noise{:.1f}_'.format(kwargs['target_noise']) \
              + 'polyak{:.3f}_'.format(kwargs['polyak']) \
              + get_lr_and_batch_size_str(**kwargs) \
              + get_iter_str(**kwargs) \
              + get_seed_str(**kwargs)

    return get_dir_path(env.unwrapped.__str__(), kwargs['agent'], param_str)

def get_naf_dir_path(env, **kwargs):
    '''
    '''

    # set parameters string
    param_str = 'dt{:.0e}_'.format(env.dt) \
              + 'gamma{:.3f}_'.format(kwargs['gamma']) \
              + get_model_arch_str(**kwargs) \
              + 'n-steps-lim{:.1e}_'.format(kwargs['n_steps_lim']) \
              + get_action_limit_str(**kwargs) \
              + 'expl-noise{:.1f}_'.format(kwargs['expl_noise_init']) \
              + 'polyak{:.3f}_'.format(kwargs['polyak']) \
              + get_lr_and_batch_size_str(**kwargs) \
              + get_iter_str(**kwargs) \
              + get_seed_str(**kwargs)

    return get_dir_path(env.unwrapped.__str__(), kwargs['agent'], param_str)

def get_vracer_dir_path(env, **kwargs):

    # set parameters string
    param_str = 'dt{:.0e}_'.format(env.dt) \
              + 'gamma{:.3f}_'.format(kwargs['gamma']) \
              + get_baseline_str(env, **kwargs) \
              + get_model_arch_str(**kwargs) \
              + 'n-steps-lim{:.1e}_'.format(kwargs['n_steps_lim']) \
              + 'expl-noise{:.1f}_'.format(kwargs['expl_noise_init']) \
              + get_lr_and_batch_size_str(**kwargs) \
              + 'policy-freq{:d}_'.format(kwargs['policy_freq']) \
              + 'cut-off{:.3f}_'.format(kwargs['cutoff_scale']) \
              + get_replay_memory_str(**kwargs) \
              + get_iter_str(**kwargs) \
              + get_seed_str(**kwargs)

    return get_dir_path(env.unwrapped.__str__(), 'vracer', param_str)

