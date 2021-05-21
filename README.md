[![Docs](https://readthedocs.org/projects/pysplice/badge/?style=flat)](https://pysplice.readthedocs.io/en/latest/)

# Splice Machine Python Package
This package contains all of the classes and functions you need to interact with Splice Machine's scale out, Hadoop on SQL RDBMS from Python. It also contains several machine learning utilities for use with Apache Spark.

## Installation Instructions: with Pip
`(sudo) pip install splicemachine`

### To include notebook utilities
`(sudo) pip install splicemachine[notebook]`

### To include statistics utilities
`(sudo) pip install splicemachine[stats]`

### To include all extras (recommended)
`(sudo) pip install splicemachine[all]`

<b>NOTE:</b> If you use zsh and plan to install extras, you must escape the brackets (`pip install splicemachine\[all\]`

## Modules
This package contains 4 main external modules. First, `splicemachine.spark.context`, which houses our Python wrapped Native Spark Datasource, as well as our External Native Spark Datasource, for use outside of the Kubernetes Cluster. Second, `splicemachine.mlflow_support` which houses our Python interface to MLManager. Lastly, `splicemachine.stats` which houses functions/classes which simplify machine learning (by providing functions like Decision Tree Visualizers, Model Evaluators etc.) and `splicemachine.notebook` which provides Jupyter Notebook specific functionality like an embedded MLFlow UI and Spark Jobs UI.

1) [`splicemachine.spark.context`](https://pysplice.readthedocs.io/en/latest/splicemachine.spark.html): Native Spark Datasource for interacting with Splice Machine from Spark
    
    1.1) [`splicemachine.spark.context.ExtPySpliceContext`](https://pysplice.readthedocs.io/en/latest/splicemachine.spark.html#splicemachine.spark.context.ExtPySpliceContext): External Native Spark Datasource for interacting with Splice Machine from Spark. Usage is mostly identical to above after instantiation (with a few extra functions available). To instantiate, you must provide the `kafkaServers` parameter pointing to the Kafka URL of the splice cluster you want to connect to. In Standalone, that url will be the default parameter of the class (`localhost:9092`)
 
 
2) [`splicemachine.mlflow_support`](https://pysplice.readthedocs.io/en/latest/splicemachine.mlflow_support.html): MLFlow wrapped MLManager interface from Python. The majority of documentation is identical to [MLflow](https://www.mlflow.org/docs/1.15.0/index.html). Additional functions and functionality are available in the docs

3) [`splicemachine.features`](https://pysplice.readthedocs.io/en/latest/splicemachine.features.html): The Python SDK entrypoint to the [Splice Machine Feature Store](https://splicemachine.com/product/feature-store/)

4) Extensions
 
    4.1) [`splicemachine.stats`](https://pysplice.readthedocs.io/en/latest/splicemachine.stats.html): houses utilities for machine learning
    
    4.2) [`splicemachine.notebooks`](https://pysplice.readthedocs.io/en/latest/splicemachine.notebook.html): houses utilities for use in Jupyter Notebooks running in the Kubernetes cloud environment

## Docs
The docs are managed py readthedocs and Sphinx. See latest docs [here](https://pysplice.readthedocs.io/en/latest/)

### Building the docs
```
cd docs
make html
```
