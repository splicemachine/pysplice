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

from setuptools import setup, find_packages

dependencies = [
    "py4j==0.10.7",
    "pytest==3.6.1",
    "six==1.11.0",
    "mlflow==1.1.0",
    "graphviz==0.8.4",
    "numpy==1.15.0",
    "pandas==0.22.0"
]
setup(
    name="splicemachine",
    version="1.0.0",
    install_requires=dependencies,
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
