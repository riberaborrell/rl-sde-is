from setuptools import setup, find_packages

import os


with open("VERSION", "r") as f:
    VERSION = f.read().strip("\n")

_dir = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(_dir, 'README.md'), 'r') as f:
    README = f.read()


setup(
    name='rl-sde-is',
    version=VERSION,
    description='Reinforcement learning algorithms for solving the importance sampling SOC problem.',
    long_description_content_type='text/markdown',
    long_description=README,
    classifiers=[],
    url='https://github.com/riberaborrell/rl-sde-is',
    #license='GNU General Public License V3',
    author='Enric Ribera Borrell',
    author_email='ribera.borrell@me.com',
    keywords=[],
    zip_safe=True,
    include_package_data=True,
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'torch',
    ],
    extras_require={
        'test': ['pytest'],
    },
)
