![Build Status](https://readthedocs.org/projects/pysplice/badge/?style=flat)


# Splice Machine Python Package
This package contains all of the classes and functions you need to interact with Splice Machine's scale out, Hadoop on SQL RDBMS from Python. It also contains several machine learning utilities for use with Apache Spark.

## Installation Instructions: with Pip
`(sudo) pip install git+https://github.com/splicemachine/pysplice`

## Modules
This package contains 4 main external modules. First, `splicemachine.spark.context`, which houses our Python wrapped Native Spark Datasource, as well as our External Native Spark Datasource, for use outside of the Kubernetes Cluster. Second, `splicemachine.mlflow_support` which houses our Python interface to MLManager. Lastly, `splicemachine.stats` which houses functions/classes which simplify machine learning (by providing functions like Decision Tree Visualizers, Model Evaluators etc.) and `splicemachine.notebook` which provides Jupyter Notebook specific functionality like an embedded MLFlow UI and Spark Jobs UI.

1) `splicemachine.spark.context`: Native Spark Datasource for interacting with Splice Machine from Spark
```
class PySpliceContext(builtins.object)
 |  PySpliceContext(sparkSession, JDBC_URL=None, _unit_testing=False)
 |  
 |  This class implements a SpliceMachineContext object (similar to the SparkContext object)
 |  
 |  Methods defined here:
 |  
 |  __init__(self, sparkSession, JDBC_URL=None, _unit_testing=False)
 |      :param JDBC_URL: (string) The JDBC URL Connection String for your Splice Machine Cluster
 |      :param sparkSession: (sparkContext) A SparkSession object for talking to Spark
 |  
 |  analyzeSchema(self, schema_name)
 |      analyze the schema
 |      :param schema_name: schema name which stats info will be collected
 |      :return:
 |  
 |  analyzeTable(self, schema_table_name, estimateStatistics=False, samplePercent=10.0)
 |      collect stats info on a table
 |      :param schema_table_name: full table name in the format of "schema.table"
 |      :param estimateStatistics:will use estimate statistics if True
 |      :param samplePercent:  the percentage or rows to be sampled.
 |      :return:
 |  
 |  bulkImportHFile(self, dataframe, schema_table_name, options)
 |      Bulk Import HFile from a dataframe into a schema.table
 |      :param dataframe: Input data
 |      :param schema_table_name: Full table name in the format of "schema.table"
 |      :param options: Dictionary of options to be passed to --splice-properties; bulkImportDirectory is required
 |  
 |  bulkImportHFileWithRdd(self, rdd, schema, schema_table_name, options)
 |      Bulk Import HFile from an rdd into a schema.table
 |      :param rdd: Input data
 |      :param schema: (StructType) The schema of the rows in the RDD
 |      :param schema_table_name: Full table name in the format of "schema.table"
 |      :param options: Dictionary of options to be passed to --splice-properties; bulkImportDirectory is required
 |  
 |  createDataFrame(self, rdd, schema)
 |      Creates a dataframe from a given rdd and schema.
 |      
 |      :param rdd: Input data
 |      :param schema: (StructType) The schema of the rows in the RDD
 |  
 |  createTable(self, dataframe, schema_table_name, primary_keys=None, create_table_options=None, to_upper=False, drop_table=False)
 |      Creates a schema.table from a dataframe
 |      :param dataframe: The Spark DataFrame to base the table off
 |      :param schema_table_name: str The schema.table to create
 |      :param primary_keys: List[str] the primary keys. Default None
 |      :param create_table_options: str The additional table-level SQL options default None
 |      :param to_upper: bool If the dataframe columns should be converted to uppercase before table creation
 |                          If False, the table will be created with lower case columns. Default False
 |      :param drop_table: bool whether to drop the table if it exists. Default False. If False and the table exists,
 |                         the function will throw an exception.
 |  
 |  createTableWithSchema(self, schema_table_name, schema, keys=None, create_table_options=None)
 |      Creates a schema.table from a schema
 |      :param schema_table_name: str The schema.table to create
 |      :param schema: (StructType) The schema that describes the columns of the table
 |      :param keys: List[str] The primary keys. Default None
 |      :param create_table_options: str The additional table-level SQL options. Default None
 |  
 |  delete(self, dataframe, schema_table_name)
 |      Delete records in a dataframe based on joining by primary keys from the data frame.
 |      Be careful with column naming and case sensitivity.
 |      
 |      :param dataframe: (DF) The dataframe you would like to delete
 |      :param schema_table_name: (string) Splice Machine Table
 |  
 |  deleteWithRdd(self, rdd, schema, schema_table_name)
 |      Delete records using an rdd based on joining by primary keys from the rdd.
 |      Be careful with column naming and case sensitivity.
 |      
 |      :param rdd: (RDD) The RDD containing the primary keys you would like to delete from the table
 |      :param schema: (StructType) The schema of the rows in the RDD
 |      :param schema_table_name: (string) Splice Machine Table
 |  
 |  df(self, sql)
 |      Return a Spark Dataframe from the results of a Splice Machine SQL Query
 |      
 |      :param sql: (string) SQL Query (eg. SELECT * FROM table1 WHERE column2 > 3)
 |      :return: A Spark DataFrame containing the results
 |  
 |  dropTable(self, schema_and_or_table_name, table_name=None)
 |      Drop a specified table.
 |      
 |      Call it like:
 |          dropTable('schemaName.tableName')
 |      Or:
 |          dropTable('schemaName', 'tableName')
 |      
 |      :param schema_and_or_table_name: (string) Pass the schema name in this param when passing the table_name param,
 |        or pass schemaName.tableName in this param without passing the table_name param
 |      :param table_name: (optional) (string) Table Name, used when schema_and_or_table_name contains only the schema name
 |  
 |  execute(self, query_string)
 |      execute a query
 |      :param query_string: (string) SQL Query (eg. SELECT * FROM table1 WHERE column2 > 3)
 |      :return:
 |  
 |  executeUpdate(self, query_string)
 |      execute a dml query:(update,delete,drop,etc)
 |      :param query_string: (string) SQL Query (eg. SELECT * FROM table1 WHERE column2 > 3)
 |      :return:
 |  
 |  export(self, dataframe, location, compression=False, replicationCount=1, fileEncoding=None, fieldSeparator=None, quoteCharacter=None)
 |      Export a dataFrame in CSV
 |      :param dataframe:
 |      :param location: Destination directory
 |      :param compression: Whether to compress the output or not
 |      :param replicationCount:  Replication used for HDFS write
 |      :param fileEncoding: fileEncoding or null, defaults to UTF-8
 |      :param fieldSeparator: fieldSeparator or null, defaults to ','
 |      :param quoteCharacter: quoteCharacter or null, defaults to '"'
 |      :return:
 |  
 |  exportBinary(self, dataframe, location, compression, e_format)
 |      Export a dataFrame in binary format
 |      :param dataframe:
 |      :param location: Destination directory
 |      :param compression: Whether to compress the output or not
 |      :param e_format: Binary format to be used, currently only 'parquet' is supported
 |      :return:
 |  
 |  getConnection(self)
 |      Return a connection to the database
 |  
 |  getSchema(self, schema_table_name)
 |      Return the schema via JDBC.
 |      
 |      :param schema_table_name: (DF) Table name
 |  
 |  insert(self, dataframe, schema_table_name, to_upper=False)
 |      Insert a dataframe into a table (schema.table).
 |      
 |      :param dataframe: (DF) The dataframe you would like to insert
 |      :param schema_table_name: (string) The table in which you would like to insert the DF
 |      :param to_upper: bool If the dataframe columns should be converted to uppercase before table creation
 |                          If False, the table will be created with lower case columns. Default False
 |  
 |  insertRdd(self, rdd, schema, schema_table_name)
 |      Insert an rdd into a table (schema.table).
 |      
 |      :param rdd: (RDD) The RDD you would like to insert
 |      :param schema: (StructType) The schema of the rows in the RDD
 |      :param schema_table_name: (string) The table in which you would like to insert the RDD
 |  
 |  insertRddWithStatus(self, rdd, schema, schema_table_name, statusDirectory, badRecordsAllowed)
 |      Insert an rdd into a table (schema.table) while tracking and limiting records that fail to insert.
 |      The status directory and number of badRecordsAllowed allow for duplicate primary keys to be
 |      written to a bad records file.  If badRecordsAllowed is set to -1, all bad records will be written
 |      to the status directory.
 |      
 |      :param rdd: (RDD) The RDD you would like to insert
 |      :param schema: (StructType) The schema of the rows in the RDD
 |      :param schema_table_name: (string) The table in which you would like to insert the dataframe
 |      :param statusDirectory The status directory where bad records file will be created
 |      :param badRecordsAllowed The number of bad records are allowed. -1 for unlimited
 |  
 |  insertWithStatus(self, dataframe, schema_table_name, statusDirectory, badRecordsAllowed)
 |      Insert a dataframe into a table (schema.table) while tracking and limiting records that fail to insert.
 |      The status directory and number of badRecordsAllowed allow for duplicate primary keys to be
 |      written to a bad records file.  If badRecordsAllowed is set to -1, all bad records will be written
 |      to the status directory.
 |      
 |      :param dataframe: (DF) The dataframe you would like to insert
 |      :param schema_table_name: (string) The table in which you would like to insert the dataframe
 |      :param statusDirectory The status directory where bad records file will be created
 |      :param badRecordsAllowed The number of bad records are allowed. -1 for unlimited
 |  
 |  internalDf(self, query_string)
 |      SQL to Dataframe translation.  (Lazy)
 |      Runs the query inside Splice Machine and sends the results to the Spark Adapter app
 |      :param query_string: (string) SQL Query (eg. SELECT * FROM table1 WHERE column2 > 3)
 |      :return: pyspark dataframe contains the result of query_string
 |  
 |  internalRdd(self, schema_table_name, column_projection=None)
 |      Table with projections in Splice mapped to an RDD.
 |      :param schema_table_name: (string) Accessed table
 |      :param column_projection: (list of strings) Names of selected columns
 |      :return RDD[Row] with the result of the projection
 |  
 |  rdd(self, schema_table_name, column_projection=None)
 |      Table with projections in Splice mapped to an RDD.
 |      :param schema_table_name: (string) Accessed table
 |      :param column_projection: (list of strings) Names of selected columns
 |      :return RDD[Row] with the result of the projection
 |  
 |  replaceDataframeSchema(self, dataframe, schema_table_name)
 |      Returns a dataframe with all column names replaced with the proper string case from the DB table
 |      :param dataframe: A dataframe with column names to convert
 |      :param schema_table_name: The schema.table with the correct column cases to pull from the database
 |  
 |  splitAndInsert(self, dataframe, schema_table_name, sample_fraction)
 |      Sample the dataframe, split the table, and insert a dataFrame into a schema.table.
 |      This corresponds to an insert into from select statement
 |      :param dataframe: Input data
 |      :param schema_table_name: Full table name in the format of "schema.table"
 |      :param sample_fraction: (float) A value between 0 and 1 that specifies the percentage of data in the dataFrame
 |          that should be sampled to determine the splits.
 |          For example, specify 0.005 if you want 0.5% of the data sampled.
 |  
 |  tableExists(self, schema_and_or_table_name, table_name=None)
 |      Check whether or not a table exists
 |      
 |      Call it like:
 |          tableExists('schemaName.tableName')
 |      Or:
 |          tableExists('schemaName', 'tableName')
 |      
 |      :param schema_and_or_table_name: (string) Pass the schema name in this param when passing the table_name param,
 |        or pass schemaName.tableName in this param without passing the table_name param
 |      :param table_name: (optional) (string) Table Name, used when schema_and_or_table_name contains only the schema name
 |  
 |  toUpper(self, dataframe)
 |      Returns a dataframe with all of the columns in uppercase
 |      :param dataframe: The dataframe to convert to uppercase
 |  
 |  truncateTable(self, schema_table_name)
 |      truncate a table
 |      :param schema_table_name: the full table name in the format "schema.table_name" which will be truncated
 |      :return:
 |  
 |  update(self, dataframe, schema_table_name)
 |      Update data from a dataframe for a specified schema_table_name (schema.table).
 |      The keys are required for the update and any other columns provided will be updated
 |      in the rows.
 |      
 |      :param dataframe: (DF) The dataframe you would like to update
 |      :param schema_table_name: (string) Splice Machine Table
 |  
 |  updateWithRdd(self, rdd, schema, schema_table_name)
 |      Update data from an rdd for a specified schema_table_name (schema.table).
 |      The keys are required for the update and any other columns provided will be updated
 |      in the rows.
 |      
 |      :param rdd: (RDD) The RDD you would like to use for updating the table
 |      :param schema: (StructType) The schema of the rows in the RDD
 |      :param schema_table_name: (string) Splice Machine Table
 |  
 |  upsert(self, dataframe, schema_table_name)
 |      Upsert the data from a dataframe into a table (schema.table).
 |      
 |      :param dataframe: (DF) The dataframe you would like to upsert
 |      :param schema_table_name: (string) The table in which you would like to upsert the RDD
 |  
 |  upsertWithRdd(self, rdd, schema, schema_table_name)
 |      Upsert the data from an RDD into a table (schema.table).
 |      
 |      :param rdd: (RDD) The RDD you would like to upsert
 |      :param schema: (StructType) The schema of the rows in the RDD
 |      :param schema_table_name: (string) The table in which you would like to upsert the RDD
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |  
 |  __dict__
 |      dictionary for instance variables (if defined)
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)
 ```
 1.1) `splicemachine.spark.context`: External Native Spark Datasource for interacting with Splice Machine from Spark. Usage is identical to above after instantiation
 ```
 class ExtPySpliceContext(PySpliceContext)
 |  ExtPySpliceContext(sparkSession, JDBC_URL=None, kafkaServers='localhost:9092', kafkaPollTimeout=20000, _unit_testing=False)
 |  
 |  This class implements a SplicemachineContext object from com.splicemachine.spark2
 |  
 |  Method resolution order:
 |      ExtPySpliceContext
 |      PySpliceContext
 |      builtins.object
 ```
 2) `splicemachine.mlflow_support`: MLFlow wrapped MLManager interface from Python. The majority of documentation is identical to [MLflow](https://www.mlflow.org/docs/1.6.0/index.html). Additional functions and functionality are below
  ```
  USAGE
     from splicemachine.mlflow_support import *
     mlflow.register_splice_context(splice) # The Native Spark Datasource (or External)
     mlflow.start_run()
     mlflow.log_param("my", "param")
     mlflow.log_metric("score", 100)
     mlflow.end_run()
You can also use syntax like this:
     with mlflow.start_run() as run:
          ...
which automatically terminates the run at the end of the block.

For the lower level client-side API, see `mlflow.client` available functions.

ADDITIONAL FUNCTIONS

register_splice_context(splice_context)
    Register a Splice Context for Spark/Database operations. This is required before most other functions become available.
    (artifact storage, for example)
    :param splice_context:  splice context to input

timer(timer_name, param=True)
    Context manager for logging
    :param timer_name:
    :param param: whether or not to log the timer as a param (default=True). If false, logs as metric.
    :return:
    
    Usage:
    mlflow.start_run()
    with mlflow.timer('run_time', param=False): # Log timer as param or metric
        mlflow.log_param("my", "param")
        mlflow.log_metric("score", 100)
        mlflow.end_run()
    You can also use the syntax like this:
    with mlflow.start_run():
        with mlflow.timer('run_time', param=False): # Log timer as param or metric
            mlflow.log_param("my", "param")
            mlflow.log_metric("score", 100)

get_run_ids_by_name(run_name, experiment_id=None)
    Gets a run id from the run name. If there are multiple runs with the same name, all run IDs are returned
    :param run_name: The name of the run
    :param experiment_id: The experiment to search in. If None, all experiments are searched
    :return: List of run ids
    
current_run_id()
    Retrieve the current run id
    :return: the current run id
    
current_exp_id()
    Retrieve the current exp id
    :return: the current experiment id
    
lp(key, value):
    Add a shortcut for logging parameters in MLFlow.
    Accessible from mlflow.lp
    :param key: key for the parameter
    :param value: value for the parameter
    
lm(key, value, step=None):
    Add a shortcut for logging metrics in MLFlow.
    Accessible from mlflow.lm
    :param key: key for the parameter
    :param value: value for the parameter
    
log_model(model, name='model'):
    Log a fitted spark pipeline/model or H2O model
    :param model: (PipelineModel or Model) is the fitted Spark Model/Pipeline or H2O model to store
        with the current run
    :param name: (str) the run relative name to store the model under

log_pipeline_stages(pipeline):
    """
    Log the pipeline stages as params for the run. Currently only Spark Pipelines are supported
    :param pipeline: fitted/unitted (spark) pipeline    
    
log_feature_transformations(unfit_pipeline):
    Log feature transformations for an unfit pipeline
    Logs --> feature movement through the pipeline.
    Currently only Spark pipelines are supported.
    :param unfit_pipeline: unfit pipeline to log
    
log_model_params(pipeline_or_model):
    Log the parameters of a fitted model or a
    model part of a fitted pipeline. Currently only Spark models/Pipelines are supported
    :param pipeline_or_model: fitted pipeline/fitted model
    
download_artifact(name, local_path, run_id=None):
    Download the artifact at the given
    run id (active default) + name
    to the local path
    :param name: (str) artifact name to load
      (with respect to the run)
    :param local_path: (str) local path to download the
      model to. This path MUST include the file extension
    :param run_id: (str) the run id to download the artifact
      from. Defaults to active run
      
get_model_name(run_id):
    Gets the model name associated with a run or None
    :param run_id:
    :return: str or None
    
load_model(run_id=None, name=None):
    Download a model from database
    and load it into Spark
    :param run_id: the id of the run to get a model from
        (the run must have an associated model with it named spark_model)
    :param name: the name of the model in the database
    
login_director(username, password):
    Authenticate into the MLManager Director
    :param username: database username
    :param password: database password
    
deploy_aws(app_name, region='us-east-2', instance_type='ml.m5.xlarge',
                run_id=None, instance_count=1, deployment_mode='replace'):
    """
    Queue Job to deploy a run to sagemaker with the
    given run id (found in MLFlow UI or through search API)
    :param run_id: the id of the run to deploy. Will default to the current
        run id.
    :param app_name: the name of the app in sagemaker once deployed
    :param region: the sagemaker region to deploy to (us-east-2,
        us-west-1, us-west-2, eu-central-1 supported)
    :param instance_type: the EC2 Sagemaker instance type to deploy on
        (ml.m4.xlarge supported)
    :param instance_count: the number of instances to load balance predictions
        on
    :param deployment_mode: the method to deploy; create=application will fail
        if an app with the name specified already exists; replace=application
        in sagemaker will be replaced with this one if app already exists;
        add=add the specified model to a prexisting application (not recommended)
    
deploy_azure(endpoint_name, resource_group, workspace, run_id=None, region='East US',
                  cpu_cores=0.1, allocated_ram=0.5, model_name=None):
    Deploy a given run to AzureML.
    :param endpoint_name: (str) the name of the endpoint in AzureML when deployed to
        Azure Container Services. Must be unique.
    :param resource_group: (str) Azure Resource Group for model. Automatically created if
        it doesn't exist.
    :param workspace: (str) the AzureML workspace to deploy the model under.
        Will be created if it doesn't exist
    :param run_id: (str) if specified, will deploy a previous run (
        must have an spark model logged). Otherwise, will default to the active run
    :param region: (str) AzureML Region to deploy to: Can be East US, East US 2, Central US,
        West US 2, North Europe, West Europe or Japan East
    :param cpu_cores: (float) Number of CPU Cores to allocate to the instance.
        Can be fractional. Default=0.1
    :param allocated_ram: (float) amount of RAM, in GB, allocated to the container.
        Default=0.5
    :param model_name: (str) If specified, this will be the name of the model in AzureML.
        Otherwise, the model name will be randomly generated.
        
deploy_db(db_schema_name, db_table_name, run_id, primary_key=None, df=None, create_model_table=False, model_cols=None, classes=None, sklearn_args={}, verbose=False, pred_threshold=None, replace=False) -> None
    Function to deploy a trained (currently Spark, Sklearn or H2O) model to the Database.
    This creates 2 tables: One with the features of the model, and one with the prediction and metadata.
    They are linked with a column called MOMENT_ID
    
    :param db_schema_name: (str) the schema name to deploy to. If None, the currently set schema will be used.
    :param db_table_name: (str) the table name to deploy to. If none, the run_id will be used for the table name(s)
    :param run_id: (str) The run_id to deploy the model on. The model associated with this run will be deployed
    
    
    OPTIONAL PARAMETERS:
    :param primary_key: (List[Tuple[str, str]]) List of column + SQL datatype to use for the primary/composite key.
                        If you are deploying to a table that already exists, this primary/composite key must exist in the table
                        If you are creating the table in this function, you MUST pass in a primary key
    :param df: (Spark or Pandas DF) The dataframe used to train the model
                NOTE: this dataframe should NOT be transformed by the model. The columns in this df are the ones
                that will be used to create the table.
    :param create_model_table: Whether or not to create the table from the dataframe. Default false. This
                                Will ONLY be used if the table does not exist and a dataframe is passed in
    :param predictor_cols: (List[str]) The columns from the table to use for the model. If None, all columns in the table
                                        will be passed to the model. If specified, the columns will be passed to the model
                                        IN THAT ORDER. The columns passed here must exist in the table.
    :param classes: (List[str]) The classes (prediction labels) for the model being deployed.
                    NOTE: If not supplied, the table will have default column names for each class
    :param sklearn_args: (dict{str: str}) Prediction options for sklearn models
                        Available key value options:
                        'predict_call': 'predict', 'predict_proba', or 'transform'
                                                                       - Determines the function call for the model
                                                                       If blank, predict will be used
                                                                       (or transform if model doesn't have predict)
                        'predict_args': 'return_std' or 'return_cov' - For Bayesian and Gaussian models
                                                                         Only one can be specified
                        If the model does not have the option specified, it will be ignored.
    :param verbose: (bool) Whether or not to print out the queries being created. Helpful for debugging
    :param pred_threshold: (double) A prediction threshold for *Keras* binary classification models
                            If the model type isn't Keras, this parameter will be ignored
                            NOTE: If the model type is Keras, the output layer has 1 node, and pred_threshold is None,
                                  you will NOT receive a class prediction, only the output of the final layer (like model.predict()).
                                  If you want a class prediction
                                  for your binary classification problem, you MUST pass in a threshold.
    :param replace: (bool) whether or not to replace a currently existing model. This param does not yet work
    
    
    This function creates the following:
    IF you are creating the table from the dataframe:
        * The model table where run_id is the run_id passed in (or the current active run_id
            This will have a column for each feature in the feature vector. It will also contain:
            USER which is the current user who made the request
            EVAL_TIME which is the CURRENT_TIMESTAMP
            the PRIMARY KEY column same as the DATA table to link predictions to rows in the table (primary key)
            PREDICTION. The prediction of the model. If the :classes: param is not filled in, this will be default values for classification models
            A column for each class of the predictor with the value being the probability/confidence of the model if applicable
    IF you are deploying to an existing table:
        * The table will be altered to include
    
    * A trigger that runs on (after) insertion to the data table that runs an INSERT into the prediction table,
        calling the PREDICT function, passing in the row of data as well as the schema of the dataset, and the run_id of the model to run
    * A trigger that runs on (after) insertion to the prediction table that calls an UPDATE to the row inserted,
        parsing the prediction probabilities and filling in proper column values
        
get_deployed_models() -> PandasDF:
    """
    Get the currently deployed models in the database
    :return: Pandas df
```
 3.1) `splicemachine.stats`: houses utilities for machine learning
```
    class DecisionTreeVisualizer(builtins.object)
     |  Visualize a decision tree, either in code like format, or graphviz
     |  
     |  Static methods defined here:
     |  
     |  add_node(dot, parent, node_hash, root, realroot=False)
     |      Traverse through the .debugString json and generate a graphviz tree
     |      :param dot: dot file object
     |      :param parent: not used currently
     |      :param node_hash: unique node id
     |      :param root: the root of tree
     |      :param realroot: whether or not it is the real root, or a recursive root
     |      :return:
     |  
     |  feature_importance(spark, model, dataset, featuresCol='features')
     |      Return a dataframe containing the relative importance of each feature
     |      :param model:
     |      :param dataframe:
     |      :param featureCol:
     |      :return: dataframe containing importance
     |  
     |  parse(lines)
     |      Lines in debug string
     |      :param lines:
     |      :return: block json
     |  
     |  replacer(string, bad, good)
     |      Replace every string in "bad" with the corresponding string in "good"
     |      :param string: string to replace in
     |      :param bad: array of strings to replace
     |      :param good: array of strings to replace with
     |      :return:
     |  
     |  tree_json(tree)
     |      Generate a JSON representation of a decision tree
     |      :param tree: tree debug string
     |      :return: json
     |  
     |  visualize(model, feature_column_names, label_names, size=None, horizontal=False, tree_name='tree', visual=False)
     |      Visualize a decision tree, either in a code like format, or graphviz
     |      :param model: the fitted decision tree classifier
     |      :param feature_column_names: (List[str]) column names for features
     |             You can access these feature names by using your VectorAssembler (in PySpark) and calling it's .getInputCols() function
     |      :param label_names: (List[str]) labels vector (below avg, above avg)
     |      :param size: tuple(int,int) The size of the graph. If unspecified, graphviz will automatically assign a size
     |      :param horizontal: (Bool) if the tree should be rendered horizontally
     |      :param tree_name: the name you would like to call the tree
     |      :param visual: bool, true if you want a graphviz pdf containing your file
     |      :return dot: The graphvis object
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
     |  ----------------------------------------------------------------------
     |
    class IndReconstructer(pyspark.ml.base.Transformer, pyspark.ml.param.shared.HasInputCol, pyspark.ml.param.shared.HasOutputCol, pyspark.ml.util.DefaultParamsReadable, pyspark.ml.util.DefaultParamsWritable)
     |  IndReconstructer(inputCol=None, outputCol=None)
     |  
     |  Transformer to reconstruct String Index from OneHotDummy Columns. This can be used as a part of a Pipeline Ojbect
     |  Follows: https://spark.apache.org/docs/latest/ml-pipeline.html#transformers
     |  :param Transformer: Inherited Class
     |  :param HasInputCol: Inherited Class
     |  :param HasOutputCol: Inherited Class
     |  :return: Transformed PySpark Dataframe With Original String Indexed Variables
     |  
     |  Method resolution order:
     |      IndReconstructer
     |      pyspark.ml.base.Transformer
     |      pyspark.ml.param.shared.HasInputCol
     |      pyspark.ml.param.shared.HasOutputCol
     |      pyspark.ml.param.Params
     |      pyspark.ml.util.Identifiable
     |      pyspark.ml.util.DefaultParamsReadable
     |      pyspark.ml.util.MLReadable
     |      pyspark.ml.util.DefaultParamsWritable
     |      pyspark.ml.util.MLWritable
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, inputCol=None, outputCol=None)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  setParams(self, inputCol=None, outputCol=None)
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from pyspark.ml.base.Transformer:
     |  
     |  serializeToBundle(self, path, dataset=None)
     |  
     |  transform(self, dataset, params=None)
     |      Transforms the input dataset with optional parameters.
     |      
     |      :param dataset: input dataset, which is an instance of :py:class:`pyspark.sql.DataFrame`
     |      :param params: an optional param map that overrides embedded params.
     |      :returns: transformed dataset
     |      
     |      .. versionadded:: 1.3.0
     |
     |  ----------------------------------------------------------------------
     |
    class MarkovChain(builtins.object)
     |  MarkovChain(transition_prob)
     |  
     |  Methods defined here:
     |  
     |  __init__(self, transition_prob)
     |      Initialize the MarkovChain instance.
     |      Parameters
     |      ----------
     |      transition_prob: dict
     |          A dict object representing the transition
     |          probabilities in Markov Chain.
     |          Should be of the form:
     |              {'state1': {'state1': 0.1, 'state2': 0.4},
     |               'state2': {...}}
     |  
     |  generate_states(self, current_state, no=10, last=True)
     |      Generates the next states of the system.
     |      Parameters
     |      ----------
     |      current_state: str
     |          The state of the current random variable.
     |      no: int
     |          The number of future states to generate.
     |      last: bool
     |          Do we want to return just the last value
     |  
     |  get_max_num_steps(self)
     |  
     |  next_state(self, current_state)
     |      Returns the state of the random variable at the next time
     |      instance.
     |      :param current_state: The current state of the system.
     |      :raises: Exception if random choice fails
     |      :return: next state
     |  
     |  rep_states(self, current_state, no=10, num_reps=10)
     |      running generate states a bunch of times and returning the final state that happens the most
     |      Arguments:
     |          current_state str -- The state of the current random variable
     |          no int -- number of time steps in the future to run
     |          num_reps int -- number of times to run the simultion forward
     |      Returns:
     |          state -- the most commonly reached state at the end of these runs
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
     |  ----------------------------------------------------------------------
     |
    class OneHotDummies(pyspark.ml.base.Transformer, pyspark.ml.param.shared.HasInputCol, pyspark.ml.param.shared.HasOutputCol, pyspark.ml.util.DefaultParamsReadable, pyspark.ml.util.DefaultParamsWritable)
     |  OneHotDummies(inputCol=None, outputCol=None)
     |  
     |  Transformer to generate dummy columns for categorical variables as a part of a preprocessing pipeline
     |  Follows: https://spark.apache.org/docs/latest/ml-pipeline.html#transformers
     |  :param Transformer: Inherited Classes
     |  :param HasInputCol: Inherited Classes
     |  :param HasOutputCol: Inherited Classes
     |  :return: pyspark DataFrame
     |  
     |  Method resolution order:
     |      OneHotDummies
     |      pyspark.ml.base.Transformer
     |      pyspark.ml.param.shared.HasInputCol
     |      pyspark.ml.param.shared.HasOutputCol
     |      pyspark.ml.param.Params
     |      pyspark.ml.util.Identifiable
     |      pyspark.ml.util.DefaultParamsReadable
     |      pyspark.ml.util.MLReadable
     |      pyspark.ml.util.DefaultParamsWritable
     |      pyspark.ml.util.MLWritable
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, inputCol=None, outputCol=None)
     |      Assigns variables to parameters passed
     |      :param inputCol: Sparse vector returned by OneHotEncoders, defaults to None
     |      :param outputCol: string base to append to output columns names, defaults to None
     |  
     |  getOutCols(self)
     |  
     |  setParams(self, inputCol=None, outputCol=None)
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from pyspark.ml.base.Transformer:
     |  
     |  serializeToBundle(self, path, dataset=None)
     |  
     |  transform(self, dataset, params=None)
     |      Transforms the input dataset with optional parameters.
     |      
     |      :param dataset: input dataset, which is an instance of :py:class:`pyspark.sql.DataFrame`
     |      :param params: an optional param map that overrides embedded params.
     |      :returns: transformed dataset
     |      
     |      .. versionadded:: 1.3.0
     |  ----------------------------------------------------------------------
     |
    class OverSampleCrossValidator(pyspark.ml.tuning.CrossValidator)
     |  OverSampleCrossValidator(estimator, estimatorParamMaps, evaluator, numFolds=3, seed=None, parallelism=3, collectSubModels=False, labelCol='label', altEvaluators=None, overSample=True)
     |  
     |  Class to perform Cross Validation model evaluation while over-sampling minority labels.
     |  Example:
     |  -------
     |  >>> from pyspark.sql.session import SparkSession
     |  >>> from pyspark.stats.classification import LogisticRegression
     |  >>> from pyspark.stats.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
     |  >>> from pyspark.stats.linalg import Vectors
     |  >>> from splicemachine.stats.stats import OverSampleCrossValidator
     |  >>> spark = SparkSession.builder.getOrCreate()
     |  >>> dataset = spark.createDataFrame(
     |  ...      [(Vectors.dense([0.0]), 0.0),
     |  ...       (Vectors.dense([0.5]), 0.0),
     |  ...       (Vectors.dense([0.4]), 1.0),
     |  ...       (Vectors.dense([0.6]), 1.0),
     |  ...       (Vectors.dense([1.0]), 1.0)] * 10,
     |  ...      ["features", "label"])
     |  >>> lr = LogisticRegression()
     |  >>> grid = ParamGridBuilder().addGrid(lr.maxIter, [0, 1]).build()
     |  >>> PRevaluator = BinaryClassificationEvaluator(metricName = 'areaUnderPR')
     |  >>> AUCevaluator = BinaryClassificationEvaluator(metricName = 'areaUnderROC')
     |  >>> ACCevaluator = MulticlassClassificationEvaluator(metricName="accuracy")
     |  >>> cv = OverSampleCrossValidator(estimator=lr, estimatorParamMaps=grid, evaluator=AUCevaluator, altEvaluators = [PRevaluator, ACCevaluator],parallelism=2,seed = 1234)
     |  >>> cvModel = cv.fit(dataset)
     |  >>> print(cvModel.avgMetrics)
     |  [(0.5, [0.5888888888888888, 0.3888888888888889]), (0.806878306878307, [0.8556863149300125, 0.7055555555555556])]
     |  >>> print(AUCevaluator.evaluate(cvModel.transform(dataset)))
     |  0.8333333333333333
     |  
     |  Method resolution order:
     |      OverSampleCrossValidator
     |      pyspark.ml.tuning.CrossValidator
     |      pyspark.ml.base.Estimator
     |      pyspark.ml.tuning._CrossValidatorParams
     |      pyspark.ml.tuning._ValidatorParams
     |      pyspark.ml.param.shared.HasSeed
     |      pyspark.ml.param.shared.HasParallelism
     |      pyspark.ml.param.shared.HasCollectSubModels
     |      pyspark.ml.param.Params
     |      pyspark.ml.util.Identifiable
     |      pyspark.ml.util.MLReadable
     |      pyspark.ml.util.MLWritable
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, estimator, estimatorParamMaps, evaluator, numFolds=3, seed=None, parallelism=3, collectSubModels=False, labelCol='label', altEvaluators=None, overSample=True)
     |      Initialize Self
     |      :param estimator: Machine Learning Model, defaults to None
     |      :param estimatorParamMaps: paramMap to search, defaults to None
     |      :param evaluator: primary model evaluation metric, defaults to None
     |      :param numFolds: number of folds to perform, defaults to 3
     |      :param seed: random state, defaults to None
     |      :param parallelism: number of threads, defaults to 1
     |      :param collectSubModels: to return submodels, defaults to False
     |      :param labelCol: target variable column label, defaults to 'label'
     |      :param altEvaluators: additional metrics to evaluate, defaults to None
     |                           If passed, the metrics of the alternate evaluators are accessed in the CrossValidatorModel.avgMetrics attribute
     |      :param overSample: Boolean: to perform oversampling of minority labels, defaults to True
     |  
     |  getAltEvaluators(self)
     |  
     |  getLabel(self)
     |  
     |  getOversample(self)
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from pyspark.ml.tuning.CrossValidator:
     |  
     |  copy(self, extra=None)
     |      Creates a copy of this instance with a randomly generated uid
     |      and some extra params. This copies creates a deep copy of
     |      the embedded paramMap, and copies the embedded and extra parameters over.
     |      
     |      :param extra: Extra parameters to copy to the new instance
     |      :return: Copy of this instance
     |      
     |      .. versionadded:: 1.4.0
     |  
     |  setCollectSubModels(self, value)
     |      Sets the value of :py:attr:`collectSubModels`.
     |  
     |  setEstimator(self, value)
     |      Sets the value of :py:attr:`estimator`.
     |      
     |      .. versionadded:: 2.0.0
     |  
     |  setEstimatorParamMaps(self, value)
     |      Sets the value of :py:attr:`estimatorParamMaps`.
     |      
     |      .. versionadded:: 2.0.0
     |  
     |  setEvaluator(self, value)
     |      Sets the value of :py:attr:`evaluator`.
     |      
     |      .. versionadded:: 2.0.0
     |  
     |  setNumFolds(self, value)
     |      Sets the value of :py:attr:`numFolds`.
     |      
     |      .. versionadded:: 1.4.0
     |  
     |  setParallelism(self, value)
     |      Sets the value of :py:attr:`parallelism`.
     |  
     |  setParams(self, estimator=None, estimatorParamMaps=None, evaluator=None, numFolds=3, seed=None, parallelism=1, collectSubModels=False)
     |      setParams(self, estimator=None, estimatorParamMaps=None, evaluator=None, numFolds=3,                  seed=None, parallelism=1, collectSubModels=False):
     |      Sets params for cross validator.
     |      
     |      .. versionadded:: 1.4.0
     |  
     |  setSeed(self, value)
     |      Sets the value of :py:attr:`seed`.
     |  
     |  write(self)
     |      Returns an MLWriter instance for this ML instance.
     |      
     |      .. versionadded:: 2.3.0
     |  
     |  ----------------------------------------------------------------------
     |  Class methods inherited from pyspark.ml.tuning.CrossValidator:
     |  
     |  read() from builtins.type
     |      Returns an MLReader instance for this class.
     |      
     |      .. versionadded:: 2.3.0
     |  ----------------------------------------------------------------------
     |
    class OverSampler(pyspark.ml.base.Transformer, pyspark.ml.param.shared.HasInputCol, pyspark.ml.param.shared.HasOutputCol, pyspark.ml.util.DefaultParamsReadable, pyspark.ml.util.DefaultParamsWritable)
     |  OverSampler(labelCol=None, strategy='auto', randomState=None)
     |  
     |  Transformer to oversample datapoints with minority labels
     |  Follows: https://spark.apache.org/docs/latest/ml-pipeline.html#transformers
     |  :param Transformer: Inherited Class
     |  :param HasInputCol: Inherited Class
     |  :param HasOutputCol: Inherited Class
     |  :return: PySpark Dataframe with labels in approximately equal ratios
     |  Example:
     |  -------
     |  >>> from pyspark.sql import functions as F
     |  >>> from pyspark.sql.session import SparkSession
     |  >>> from pyspark.stats.linalg import Vectors
     |  >>> from splicemachine.stats.stats import OverSampler
     |  >>> spark = SparkSession.builder.getOrCreate()
     |  >>> df = spark.createDataFrame(
     |  ...      [(Vectors.dense([0.0]), 0.0),
     |  ...       (Vectors.dense([0.5]), 0.0),
     |  ...       (Vectors.dense([0.4]), 1.0),
     |  ...       (Vectors.dense([0.6]), 1.0),
     |  ...       (Vectors.dense([1.0]), 1.0)] * 10,
     |  ...      ["features", "Class"])
     |  >>> df.groupBy(F.col("Class")).count().orderBy("count").show()
     |  +-----+-----+
     |  |Class|count|
     |  +-----+-----+
     |  |  0.0|   20|
     |  |  1.0|   30|
     |  +-----+-----+
     |  >>> oversampler = OverSampler(labelCol = "Class", strategy = "auto")
     |  >>> oversampler.transform(df).groupBy("Class").count().show()
     |  +-----+-----+
     |  |Class|count|
     |  +-----+-----+
     |  |  0.0|   29|
     |  |  1.0|   30|
     |  +-----+-----+
     |  
     |  Method resolution order:
     |      OverSampler
     |      pyspark.ml.base.Transformer
     |      pyspark.ml.param.shared.HasInputCol
     |      pyspark.ml.param.shared.HasOutputCol
     |      pyspark.ml.param.Params
     |      pyspark.ml.util.Identifiable
     |      pyspark.ml.util.DefaultParamsReadable
     |      pyspark.ml.util.MLReadable
     |      pyspark.ml.util.DefaultParamsWritable
     |      pyspark.ml.util.MLWritable
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, labelCol=None, strategy='auto', randomState=None)
     |      Initialize self
     |      :param labelCol: Label Column name, defaults to None
     |      :param strategy: defaults to "auto", strategy to resample the dataset:
     |                      • Only currently supported for "auto" Corresponds to random samples with repleaement
     |      :param randomState: sets the seed of sample algorithm
     |  
     |  setParams(self, labelCol=None, strategy='auto')
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from pyspark.ml.base.Transformer:
     |  
     |  serializeToBundle(self, path, dataset=None)
     |  
     |  transform(self, dataset, params=None)
     |      Transforms the input dataset with optional parameters.
     |      
     |      :param dataset: input dataset, which is an instance of :py:class:`pyspark.sql.DataFrame`
     |      :param params: an optional param map that overrides embedded params.
     |      :returns: transformed dataset
     |      
     |      .. versionadded:: 1.3.0
     |  
     |  ----------------------------------------------------------------------
     |  Static methods inherited from pyspark.ml.base.Transformer:
     |  
     |  deserializeFromBundle(path)
     |  ----------------------------------------------------------------------
     |
    class Rounder(pyspark.ml.base.Transformer, pyspark.ml.param.shared.HasInputCol, pyspark.ml.param.shared.HasOutputCol, pyspark.ml.util.DefaultParamsReadable, pyspark.ml.util.DefaultParamsWritable)
     |  Rounder(predictionCol='prediction', labelCol='label', clipPreds=True, maxLabel=None, minLabel=None)
     |  
     |  Transformer to round predictions for ordinal regression
     |  Follows: https://spark.apache.org/docs/latest/ml-pipeline.html#transformers
     |  :param Transformer: Inherited Class
     |  :param HasInputCol: Inherited Class
     |  :param HasOutputCol: Inherited Class
     |  :return: Transformed Dataframe with rounded predictionCol
     |  Example:
     |  --------
     |  >>> from pyspark.sql.session import SparkSession
     |  >>> from splicemachine.stats.stats import Rounder
     |  >>> spark = SparkSession.builder.getOrCreate()
     |  >>> dataset = spark.createDataFrame(
     |  ...      [(0.2, 0.0),
     |  ...       (1.2, 1.0),
     |  ...       (1.6, 2.0),
     |  ...       (1.1, 0.0),
     |  ...       (3.1, 0.0)],
     |  ...      ["prediction", "label"])
     |  >>> dataset.show()
     |  +----------+-----+
     |  |prediction|label|
     |  +----------+-----+
     |  |       0.2|  0.0|
     |  |       1.2|  1.0|
     |  |       1.6|  2.0|
     |  |       1.1|  0.0|
     |  |       3.1|  0.0|
     |  +----------+-----+
     |  >>> rounder = Rounder(predictionCol = "prediction", labelCol = "label", clipPreds = True)
     |  >>> rounder.transform(dataset).show()
     |  +----------+-----+
     |  |prediction|label|
     |  +----------+-----+
     |  |       0.0|  0.0|
     |  |       1.0|  1.0|
     |  |       2.0|  2.0|
     |  |       1.0|  0.0|
     |  |       2.0|  0.0|
     |  +----------+-----+
     |  >>> rounderNoClip = Rounder(predictionCol = "prediction", labelCol = "label", clipPreds = False)
     |  >>> rounderNoClip.transform(dataset).show()
     |  +----------+-----+
     |  |prediction|label|
     |  +----------+-----+
     |  |       0.0|  0.0|
     |  |       1.0|  1.0|
     |  |       2.0|  2.0|
     |  |       1.0|  0.0|
     |  |       3.0|  0.0|
     |  +----------+-----+
     |  
     |  Method resolution order:
     |      Rounder
     |      pyspark.ml.base.Transformer
     |      pyspark.ml.param.shared.HasInputCol
     |      pyspark.ml.param.shared.HasOutputCol
     |      pyspark.ml.param.Params
     |      pyspark.ml.util.Identifiable
     |      pyspark.ml.util.DefaultParamsReadable
     |      pyspark.ml.util.MLReadable
     |      pyspark.ml.util.DefaultParamsWritable
     |      pyspark.ml.util.MLWritable
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, predictionCol='prediction', labelCol='label', clipPreds=True, maxLabel=None, minLabel=None)
     |      initialize self
     |      :param predictionCol: column containing predictions, defaults to "prediction"
     |      :param labelCol: column containing labels, defaults to "label"
     |      :param clipPreds: clip all predictions above a specified maximum value
     |      :param maxLabel: optional: the maximum value for the prediction column, otherwise uses the maximum of the labelCol, defaults to None
     |      :param minLabel: optional: the minimum value for the prediction column, otherwise uses the maximum of the labelCol, defaults to None
     |  
     |  setParams(self, predictionCol='prediction', labelCol='label')
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from pyspark.ml.base.Transformer:
     |  
     |  serializeToBundle(self, path, dataset=None)
     |  
     |  transform(self, dataset, params=None)
     |      Transforms the input dataset with optional parameters.
     |      
     |      :param dataset: input dataset, which is an instance of :py:class:`pyspark.sql.DataFrame`
     |      :param params: an optional param map that overrides embedded params.
     |      :returns: transformed dataset
     |      
     |      .. versionadded:: 1.3.0
     |  
     |  ----------------------------------------------------------------------
     |  Static methods inherited from pyspark.ml.base.Transformer:
     |  
     |  deserializeFromBundle(path)
     |  ----------------------------------------------------------------------
     |
     class SpliceBaseEvaluator(builtins.object)
     |  SpliceBaseEvaluator(spark, evaluator, supported_metrics, predictionCol='prediction', labelCol='label')
     |  
     |  Base ModelEvaluator
     |  
     |  Methods defined here:
     |  
     |  __init__(self, spark, evaluator, supported_metrics, predictionCol='prediction', labelCol='label')
     |      Constructor for SpliceBaseEvaluator
     |      :param spark: spark from zeppelin
     |      :param evaluator: evaluator class from spark
     |      :param supported_metrics: supported metrics list
     |      :param predictionCol: prediction column
     |      :param labelCol: label column
     |  
     |  get_results(self, as_dict=False)
     |      Get Results
     |      :param dict: whether to get results in a dict or not
     |      :return: dictionary
     |  
     |  input(self, predictions_dataframe)
     |      Input a dataframe
     |      :param ev: evaluator class
     |      :param predictions_dataframe: input df
     |      :return: none
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
     |  ----------------------------------------------------------------------
     |
    class SpliceBinaryClassificationEvaluator(SpliceBaseEvaluator)
     |  SpliceBinaryClassificationEvaluator(spark, predictionCol='prediction', labelCol='label', confusion_matrix=True)
     |  
     |  Base ModelEvaluator
     |  
     |  Method resolution order:
     |      SpliceBinaryClassificationEvaluator
     |      SpliceBaseEvaluator
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, spark, predictionCol='prediction', labelCol='label', confusion_matrix=True)
     |      Constructor for SpliceBaseEvaluator
     |      :param spark: spark from zeppelin
     |      :param evaluator: evaluator class from spark
     |      :param supported_metrics: supported metrics list
     |      :param predictionCol: prediction column
     |      :param labelCol: label column
     |  
     |  input(self, predictions_dataframe)
     |      Evaluate actual vs Predicted in a dataframe
     |      :param predictions_dataframe: the dataframe containing the label and the predicition
     |  
     |  plotROC(self, fittedEstimator, ax)
     |      Plots the receiver operating characteristic curve for the trained classifier
     |      :param fittedEstimator: fitted logistic regression model
     |      :param ax: matplotlib axis object
     |      :return: axis with ROC plot
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from SpliceBaseEvaluator:
     |  
     |  get_results(self, as_dict=False)
     |      Get Results
     |      :param dict: whether to get results in a dict or not
     |      :return: dictionary
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from SpliceBaseEvaluator:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
     |  ----------------------------------------------------------------------
     |
    class SpliceMultiClassificationEvaluator(SpliceBaseEvaluator)
     |  SpliceMultiClassificationEvaluator(spark, predictionCol='prediction', labelCol='label')
     |  
     |  Base ModelEvaluator
     |  
     |  Method resolution order:
     |      SpliceMultiClassificationEvaluator
     |      SpliceBaseEvaluator
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, spark, predictionCol='prediction', labelCol='label')
     |      Constructor for SpliceBaseEvaluator
     |      :param spark: spark from zeppelin
     |      :param evaluator: evaluator class from spark
     |      :param supported_metrics: supported metrics list
     |      :param predictionCol: prediction column
     |      :param labelCol: label column
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from SpliceBaseEvaluator:
     |  
     |  get_results(self, as_dict=False)
     |      Get Results
     |      :param dict: whether to get results in a dict or not
     |      :return: dictionary
     |  
     |  input(self, predictions_dataframe)
     |      Input a dataframe
     |      :param ev: evaluator class
     |      :param predictions_dataframe: input df
     |      :return: none
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from SpliceBaseEvaluator:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    class SpliceRegressionEvaluator(SpliceBaseEvaluator)
     |  SpliceRegressionEvaluator(spark, predictionCol='prediction', labelCol='label')
     |  
     |  Splice Regression Evaluator
     |  
     |  Method resolution order:
     |      SpliceRegressionEvaluator
     |      SpliceBaseEvaluator
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, spark, predictionCol='prediction', labelCol='label')
     |      Constructor for SpliceBaseEvaluator
     |      :param spark: spark from zeppelin
     |      :param evaluator: evaluator class from spark
     |      :param supported_metrics: supported metrics list
     |      :param predictionCol: prediction column
     |      :param labelCol: label column
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from SpliceBaseEvaluator:
     |  
     |  get_results(self, as_dict=False)
     |      Get Results
     |      :param dict: whether to get results in a dict or not
     |      :return: dictionary
     |  
     |  input(self, predictions_dataframe)
     |      Input a dataframe
     |      :param ev: evaluator class
     |      :param predictions_dataframe: input df
     |      :return: none
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from SpliceBaseEvaluator:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)

FUNCTIONS
    best_fit_distribution(data, col_name, bins, ax)
        Model data by finding best fit distribution to data
        :param data: DataFrame with one column containing the feature whose distribution is to be investigated
        :param col_name: column name for feature
        :param bins: number of bins to use in generating the histogram of this data
        :param ax: axis to plot histogram on
        :return: (best_distribution.name, best_params, best_sse)
            best_distribution.name: string of the best distribution name
            best_params: parameters for this distribution
            best_sse: sum of squared errors for this distribution against the empirical pdf
    
    estimateCovariance(df, features_col='features')
        Compute the covariance matrix for a given dataframe.
            Note: The multi-dimensional covariance array should be calculated using outer products.  Don't forget to normalize the data by first subtracting the mean.
        :param df: PySpark dataframe
        :param features_col: name of the column with the features, defaults to 'features'
        :return: np.ndarray: A multi-dimensional array where the number of rows and columns both equal the length of the arrays in the input dataframe.
    
    get_confusion_matrix(spark, TP, TN, FP, FN)
        function that shows you a device called a confusion matrix... will be helpful when evaluating.
        It allows you to see how well your model performs
        :param TP: True Positives
        :param TN: True Negatives
        :param FP: False Positives
        :param FN: False Negatives
    
    get_string_pipeline(df, cols_to_exclude, steps=['StringIndexer', 'OneHotEncoder', 'OneHotDummies'])
        Generates a list of preprocessing stages
        :param df: DataFrame including only the training data
        :param cols_to_exclude: Column names we don't want to to include in the preprocessing (i.e. SUBJECT/ target column)
        :param stages: preprocessing steps to take
        :return:  (stages, Numeric_Columns)
            stages: list of pipeline stages to be used in preprocessing
            Numeric_Columns: list of columns that contain numeric features
    
    inspectTable(spliceMLCtx, sql, topN=5)
        Inspect the values of the columns of the table (dataframe) returned from the sql query
        :param spliceMLCtx: SpliceMLContext
        :param sql: sql string to execute
        :param topN: the number of most frequent elements of a column to return, defaults to 5
    
    make_pdf(dist, params, size=10000)
        Generate distributions's Probability Distribution Function
        :param dist: scipy.stats distribution object: https://docs.scipy.org/doc/scipy/reference/stats.html
        :param params: distribution parameters
        :param size: how many data points to generate , defaults to 10000
        :return: series of probability density function for this distribution
    
    pca_with_scores(df, k=10)
        Computes the top `k` principal components, corresponding scores, and all eigenvalues.
        Note:
            All eigenvalues should be returned in sorted order (largest to smallest). `eigh` returns
            each eigenvectors as a column.  This function should also return eigenvectors as columns.
        :param df:  A Spark dataframe with a 'features' column, which (column) consists of DenseVectors.
        :param k: The number of principal components to return., defaults to 10
        :return:(eigenvectors, `RDD` of scores, eigenvalues).
            Eigenvectors: multi-dimensional array where the number of
            rows equals the length of the arrays in the input `RDD` and the number of columns equals`k`.
            `RDD` of scores: has the same number of rows as `data` and consists of arrays of length `k`.
            Eigenvalues is an array of length d (the number of features).
    
    postprocessing_pipeline(df, cols_to_exclude)
        Assemble postprocessing pipeline to reconstruct original categorical indexed values from OneHotDummy Columns
        :param df: DataFrame Including the original string Columns
        :param cols_to_exclude: list of columns to exclude
        :return: (reconstructers, String_Columns)
            reconstructers: list of IndReconstructer stages
            String_Columns: list of columns that are being reconstructed
    
    reconstructPCA(sql, df, pc, mean, std, originalColumns, fits, pcaColumn='pcaFeatures')
        Reconstruct data from lower dimensional space after performing PCA
        :param sql: SQLContext
        :param df: PySpark DataFrame: inputted PySpark DataFrame
        :param pc: numpy.ndarray: principal components projected onto
        :param mean: numpy.ndarray: mean of original columns
        :param std: numpy.ndarray: standard deviation of original columns
        :param originalColumns: list: original column names
        :param fits: fits of features returned from best_fit_distribution
        :param pcaColumn: column in df that contains PCA features, defaults to 'pcaFeatures'
        :return: dataframe containing reconstructed data
    
    varianceExplained(df, k=10)
        returns the proportion of variance explained by `k` principal componenets. Calls the above PCA procedure
        :param df: PySpark DataFrame
        :param k: number of principal components , defaults to 10
        :return: (proportion, principal_components, scores, eigenvalues)
    
    vector_assembler_pipeline(df, columns, doPCA=False, k=10)
        After preprocessing String Columns, this function can be used to assemble a feature vector to be used for learning
        creates the following stages: VectorAssembler -> Standard Scalar [{ -> PCA}]
        :param df: DataFrame containing preprocessed Columns
        :param columns: list of Column names of the preprocessed columns
        :param doPCA:  Do you want to do PCA as part of the vector assembler? defaults to False
        :param k:  Number of Principal Components to use, defaults to 10
        :return: List of vector assembling stages
```
 3.1) `splicemachine.stats`: houses utilities for use in Jupyter Notebooks running in the Kubernetes cloud environment
FUNCTIONS
    get_mlflow_ui()
    
    get_spark_ui(port=None, spark_session=None)
    
    hide_toggle(toggle_next=False)
        Function to add a toggle at the bottom of Jupyter Notebook cells to allow the entire cell to be collapsed.
        :param toggle_next: Bool determine if the toggle should affect the current cell or the next cell
        Usage: from splicemachine.stats.utilities import hide_toggle
               hide_toggle()
```
