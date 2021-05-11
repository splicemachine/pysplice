Getting Started
===============

K8s Install
-----------

If you are running inside of the Splice Machine Cloud Service in a Jupyter Notebook, MLManager will already be installed for you. If you'd like to install it (or upgrade it), you can install from git with

.. code-block:: sh

    [sudo] pip install [--upgrade] splicemachine

External Installation
---------------------

If you would like to install outside of the K8s cluster (and use the ExtPySpliceContext), you can install the stable build with

.. code-block:: sh

    [sudo] pip install [--upgrade] splicemachine

Package Extras
---------------------

The splicemachine pypi package has 2 extra installs, `stats` and `notebook`. These include extra dependencies for usage
with build-in ML/Statistics functionality and extra jupyter specific functionality (like Feature Store Feature search)

To install them, you can install with the standard extra syntax for Pypi. If you'd like both (recommended), you can run

.. code-block:: sh

    [sudo] pip install [--upgrade] splicemachine[all]

If you are using `zsh` you must escape the package extra with

.. code-block:: sh

    [sudo] pip install [--upgrade] splicemachine\[all\]

Usage 
-----

This section covers importing and instantiating the Native Spark DataSource

.. tabs::
   
    .. tab:: Native Spark DataSource

        To use the Native Spark DataSource inside of the `cloud service<https://cloud.splicemachine.io/register?utm_source=pydocs&utm_medium=header&utm_campaign=sandbox>`_., first create a Spark Session and then import your PySpliceContext

        .. code-block:: Python

         from pyspark.sql import SparkSession
         from splicemachine.spark import PySpliceContext
         from splicemachine.mlflow_support import * # Connects your MLflow session automatically
         from splicemachine.features import FeatureStore # Splice Machine Feature Store

         spark = SparkSession.builder.getOrCreate()
         splice = PySpliceContext(spark) # The Native Spark Datasource (PySpliceContext) takes a Spark Session
         fs = FeatureStore(splice) # Create your Feature Store
         mlflow.register_splice_context(splice) # Gives mlflow native DB connection
         mlflow.register_feature_store(fs) # Tracks Feature Store work in Mlflow automatically


    .. tab:: External Native Spark DataSource

        To use the External Native Spark DataSource, create a Spark Session with your external Jars configured. Then, import your ExtPySpliceContext and set the necessary parameters.
        Once created, the functionality is identical to the internal Native Spark Datasource (PySpliceContext)
    
        .. code-block:: Python

         from pyspark.sql import SparkSession
         from splicemachine.spark import ExtPySpliceContext
         from splicemachine.mlflow_support import * # Connects your MLflow session automatically
         from splicemachine.features import FeatureStore # Splice Machine Feature Store

         spark = SparkSession.builder.config('spark.jars', '/path/to/splice_spark2-3.0.0.1962-SNAPSHOT-shaded.jar').config('spark.driver.extraClassPath', 'path/to/Splice/jars/dir/*').getOrCreate()
         JDBC_URL = '' #Set your JDBC URL here. You can get this from the Cloud Manager UI. Make sure to append ';user=<USERNAME>;password=<PASSWORD>' after ';ssl=basic' so you can authenticate in
         # The ExtPySpliceContext communicates with the database via Kafka
         kafka_server = 'kafka-broker-0-' + JDBC_URL.split('jdbc:splice://jdbc-')[1].split(':1527')[0] + ':19092' # Formatting kafka URL from JDBC
         splice = ExtPySpliceContext(spark, JDBC_URL=JDBC_URL, kafkaServers=kafka_server)

         fs = FeatureStore(splice) # Create your Feature Store
         mlflow.register_splice_context(splice) # Gives mlflow native DB connection
         mlflow.register_feature_store(fs) # Tracks Feature Store work in Mlflow automatically
