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

setup(
    name="splicemachine",
    version="2.2.0",
    install_requires=DEPENDENCIES,
    packages=find_packages(),
    license='Apache License, Version 2.0',
    long_description=open('README.md').read(),
    author="Splice Machine, Inc.",
    author_email="abaveja@splicemachine.com",
    description="This package contains all of the classes and functions you need to interact "
                "with Splice Machine's scale out, Hadoop on SQL RDBMS from Python. It also contains"
                " several machine learning utilities for use with Apache Spark.",
    url="https://github.com/splicemachine/pysplice/"
)
