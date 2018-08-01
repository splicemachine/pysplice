# Copyright (c) 2012 - 2017 Splice Machine, Inc.
#
# This file is part of Splice Machine.
# Splice Machine is free software: you can redistribute it and/or modify it under the terms of the
# GNU Affero General Public License as published by the Free Software Foundation, either
# version 3, or (at your option) any later version.
# Splice Machine is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License along with
# Splice Machine. If not, see <http://www.gnu.org/licenses/>.

from setuptools import setup

dependencies = [
    "atomicwrites==1.1.5",
    "attrs==18.1.0",
    "more-itertools==4.2.0",
    "pluggy==0.6.0",
    "py==1.5.3",
    "py4j==0.10.7",
    "pyspark==2.3.1",
    "pytest==3.6.1",
    "six==1.11.0"
]
setup(
    name="splicemachine",
    version="0.2.2",
    install_requires=dependencies,
    packages=['splicemachine'],
)
