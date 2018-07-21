from setuptools import setup, find_packages
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
