"""
 Copyright (c) 2024, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: Apache License 2.0
 For full license text, see the LICENSE file in the repo root or https://www.apache.org/licenses/LICENSE-2.0
"""

import os
from setuptools import setup, find_packages, find_namespace_packages


# with open("requirements.txt") as f:
#     requirements = f.read().splitlines()
requirements = None
setup(
    name='actionstudio',
    version='v2.0',
    author="The xLAM team, including Jianguo Zhang, Thai Hoang, Ming Zhu, Zuxin Liu, and others, with support from all our contributors, past and future.",
    description="A Lightweight Framework for Agentic Data and Training of Large Action Models",
    license="Apache 2.0 License",
    packages=find_packages(),
    include_package_data=True,
    package_data={"": ["src/*"]},
    install_requires=requirements, # Your dependencies here
)