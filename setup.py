"""
 Copyright (c) 2024, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: Apache License 2.0
 For full license text, see the LICENSE file in the repo root or https://www.apache.org/licenses/LICENSE-2.0
"""

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='xlam',
    version='0.1',
    author="The Salesforce AI Research xLAM Team (past and future) with the help of all the contributors",
    description="State-of-the-art library for AI Agent whole pipeline",
    license="Apache 2.0 License",
    url="https://github.com/SalesforceAIResearch/xLAM",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
)
