Getting Started
===============

K8s Install
-----------

If you are running inside of the Splice Machine Cloud Service in a Jupyter Notebook, MLManager will already be installed for you. If you'd like to install it (or upgrade it), you can install from git with

.. code-block:: sh

    pip install [--upgrade] git+https://www.github.com/splicemachine/pysplice@<RELEASE>

External Installation
---------------------

If you would like to install outside of the K8s cluster (and use the ExtPySpliceContext), you can install the stable build with

.. code-block:: sh

    sudo pip install git+http://www.github.com/splicemachine/pysplice@2.3.0-k8

Or latest with

.. code-block:: sh

    sudo pip install git+http://www.github.com/splicemachine/pysplice

Usage 
-----

This section covers importing and instantiating the Native Spark DataSource

.. tabs::
   
    .. tab:: Native Spark DataSource

        To use the Native Spark DataSource inside of the `cloud service<https://cloud.splicemachine.io/register?utm_source=pydocs&utm_medium=header&utm_campaign=sandbox>`_., first create a Spark Session and then import your PySpliceContext

        .. code-block:: Python

         from pyspark.sql import SparkSession
         from splicemachine.spark import PySpliceContext
         spark = SparkSession.builder.getOrCreate()
         splice = PySpliceContext(spark)

    .. tab:: External Native Spark DataSource

        To use the External Native Spark DataSource, create a Spark Session with your external Jars configured. Then, import your ExtPySpliceContext and set the necessary parameters
    
        .. code-block:: Python

         from pyspark.sql import SparkSession
         from splicemachine.spark import ExtPySpliceContext
         spark = SparkSession.builder.config('spark.jars', '/path/to/splice_spark2-3.0.0.1962-SNAPSHOT-shaded.jar').config('spark.driver.extraClassPath', 'path/to/Splice/jars/dir/*').getOrCreate()
         JDBC_URL = '' #Set your JDBC URL here. You can get this from the Cloud Manager UI. Make sure to append ';user=<USERNAME>;password=<PASSWORD>' after ';ssl=basic' so you can authenticate in
         kafka_server = 'kafka-broker-0-' + JDBC_URL.split('jdbc:splice://jdbc-')[1].split(':1527')[0] + ':19092' # Formatting kafka URL from JDBC 
         splice = ExtPySpliceContext(spark, JDBC_URL=JDBC_URL, kafkaServers=kafka_server)
