# coding=utf-8
# Copyright 2018 The MACI Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import codecs
from os import path
from setuptools import find_packages
from setuptools import setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with codecs.open(path.join(here, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

install_requires = ['six >= 1.10.0']
test_requirements = ['six >= 1.10.0', 'absl-py >= 0.1.6']

setup(
    name='maci',
    version='0.0.6',
    include_package_data=True,
    packages=find_packages(exclude=['docs']),  # Required
    # package_data={'testdata': ['testdata/']},
    install_requires=install_requires,
    extras_require={  # Optional
        'tf': ['tensorflow >= 1.7'],
        'gym': ['gym'],
        'keras': ['keras']
    },
    tests_require=test_requirements,
    description='MACI: A FrameWork for Multi-agent Reinforcement Learning',
    long_description=long_description,
    classifiers=[  # Optional
        'Development Status :: 0 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',

        # Pick your license as you wish
        'License :: OSI Approved :: Apache Software License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        # 'Programming Language :: Python :: 2.7',
        # 'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',

        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',

    ],
    license='Apache 2.0',
    keywords='multiagent reinforcement learning'
)
