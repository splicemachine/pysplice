"""
Copyright 2018 Splice Machine, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from setuptools import find_packages, setup

REQUIREMENTS_FILE = "requirements.txt"

with open(REQUIREMENTS_FILE, "r") as dependencies_file:
    DEPENDENCIES = dependencies_file.readlines()

with open('notebook_requirements.txt', 'r') as fs_deps:
    NOTEBOOK_DEPS = fs_deps.readlines()

with open('stats_requirements.txt', 'r') as fs_deps:
    STATS_DEPS = fs_deps.readlines()

setup(
    name="splicemachine",
    version="2.9.0.dev0",
    install_requires=DEPENDENCIES,
    extras_require={
        'notebook': NOTEBOOK_DEPS,
        'stats': STATS_DEPS,
        'all': NOTEBOOK_DEPS + STATS_DEPS
    },
    packages=find_packages(),
    license='Apache License, Version 2.0',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author="Splice Machine, Inc.",
    author_email="abaveja@splicemachine.com",
    description="This package contains all of the classes and functions you need to interact "
                "with Splice Machine's scale out, real-time, ACID compliant ML Engine. It contains "
                "several machine learning utilities for use with Apache Spark, a managed MLFlow client and a "
                "Managed Feature Store client.",
    url="https://github.com/splicemachine/pysplice/"
)
