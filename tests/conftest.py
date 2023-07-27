import numpy as np
import pytest

def pytest_addoption(parser):
    parser.addoption(
        '--seed',
        dest='seed',
        type=int,
        help='Set the seed for RandomState',
    )
    parser.addoption(
        '--d',
        dest='d',
        type=int,
        default=2,
        help='Set the dimension d. Default: 2',
    )
    parser.addoption(
        '--alpha',
        dest='alpha',
        type=float,
        default=1.,
        help='Set nd double well barrier height. Default: 1.',
    )
    parser.addoption(
        '--beta',
        dest='beta',
        type=float,
        default=1.,
        help='Set the beta parameter. Default: 1.',
    )
    parser.addoption(
        '--dt',
        dest='dt',
        type=float,
        default=0.005,
        help='Set dt. Default: 0.005',
    )
    parser.addoption(
        '--h-state',
        dest='h_state',
        type=float,
        default=0.1,
        help='Set state discretization step',
    )
    parser.addoption(
        '--h-action',
        dest='h_action',
        type=float,
        default=0.1,
        help='Set action discretization step',
    )
    parser.addoption(
        '--batch-size',
        dest='batch_size',
        type=int,
        default=10,
        help='Set batch size. Default: 10',
    )

@pytest.fixture(scope='session')
def seed(request):
    return request.config.getoption('seed')

@pytest.fixture(scope='session')
def d(request):
    return request.config.getoption('d')

@pytest.fixture(scope='session')
def alpha(request):
    return request.config.getoption('alpha')

@pytest.fixture(scope='session')
def beta(request):
    return request.config.getoption('beta')

@pytest.fixture(scope='session')
def dt(request):
    return request.config.getoption('dt')

@pytest.fixture(scope='session')
def h_state(request):
    return request.config.getoption('h_state')

@pytest.fixture(scope='session')
def h_action(request):
    return request.config.getoption('h_action')

@pytest.fixture(scope='session')
def batch_size(request):
    return request.config.getoption('batch_size')
