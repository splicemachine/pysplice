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
    "atomicwrites==1.1.5",
    "attrs==18.1.0",
    "more-itertools==4.2.0",
    "pluggy==0.6.0",
    "py==1.5.3",
    "py4j==0.10.7",
    "pytest==3.6.1",
    "six==1.11.0",
    "mlflow==0.4.2",
    "graphviz==0.8.4",
    "numpy==1.15.0",
    "h2o_pysparkling_2.2",
    "pandas==0.23.4"
]
setup(
    name="splicemachine",
    version="0.2.5",
    install_requires=dependencies,
    packages=find_packages(),
)
