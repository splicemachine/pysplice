# Splice Machine Python Package
This package contains all of the classes and functions you need to interact with Splice Machine's scale out, Hadoop on SQL RDBMS from Python. It also contains several machine learning utilities for use with Apache Spark.

## Installation Instructions: with Pip
`(sudo) pip install git+https://github.com/splicemachine/pysplice`

## Modules
This package contains three main external modules. First, `splicemachine.spark.context`, which houses our native spark datasource from Python. Second, `splicemachine.ml.management` which houses our Python interface to MLManager. Lastly, `splicemachine.ml.utilities` which houses functions/classes which simplify machine learning (by providing functions like Decision Tree Visualizers, Model Evaluators etc.)

1) `splicemachine.spark.context`: Native Spark Datasource for interacting with Splice Machine from Spark
```
class PySpliceContext(builtins.object)
     |  This class implements a SpliceMachineContext object (similar to the SparkContext object)
     |
     |  Methods defined here:
     |
     |  __init__(self, JDBC_URL, sparkSession, _unit_testing=False)
     |      :param JDBC_URL: (string) The JDBC URL Connection String for your Splice Machine Cluster
     |      :param sparkSession: (sparkContext) A SparkSession object for talking to Spark
     |
     |  analyzeSchema(self, schema_name)
     |      analyze the schema
     |      :param schema_name: schema name which stats info will be collected
     |      :return:
     |
     |  analyzeTable(self, schema_table_name, estimateStatistics=False, samplePercent=0.1)
     |      collect stats info on a table
     |      :param schema_table_name: full table name in the format of "schema.table"
     |      :param estimateStatistics:will use estimate statistics if True
     |      :param samplePercent:  the percentage or rows to be sampled.
     |      :return:
     |
     |  delete(self, dataframe, schema_table_name)
     |      Delete records in a dataframe based on joining by primary keys from the data frame.
     |      Be careful with column naming and case sensitivity.
     |
     |      :param dataframe: (DF) The dataframe you would like to delete
     |      :param schema_table_name: (string) Splice Machine Table
     |
     |  df(self, sql)
     |      Return a Spark Dataframe from the results of a Splice Machine SQL Query
     |
     |      :param sql: (string) SQL Query (eg. SELECT * FROM table1 WHERE column2 > 3)
     |      :return: A Spark DataFrame containing the results
     |
     |  dropTable(self, schema_table_name)
     |      Drop a specified table.
     |
     |      :param schema_table_name: (optional) (string) schemaName.tableName
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
     |  exportBinary(self, dataframe, location, compression, format)
     |      Export a dataFrame in binary format
     |      :param dataframe:
     |      :param location: Destination directory
     |      :param compression: Whether to compress the output or not
     |      :param format: Binary format to be used, currently only 'parquet' is supported
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
     |  insert(self, dataframe, schema_table_name)
     |      Insert a dataframe into a table (schema.table).
     |
     |      :param dataframe: (DF) The dataframe you would like to insert
     |      :param schema_table_name: (string) The table in which you would like to insert the RDD
     |
     |  internalDf(self, query_string)
     |      SQL to Dataframe translation.  (Lazy)
     |      Runs the query inside Splice Machine and sends the results to the Spark Adapter app
     |      :param query_string: (string) SQL Query (eg. SELECT * FROM table1 WHERE column2 > 3)
     |      :return: pyspark dataframe contains the result of query_string
     |
     |  tableExists(self, schema_table_name)
     |      Check whether or not a table exists
     |
     |      :param schema_table_name: (string) Table Name
     |
     |  toUpper(self, dataframe)
     |      Returns a dataframe with all uppercase column names
     |      :param dataframe: A dataframe with column names to convert to uppercase
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
     |      :return:
     |
     |  upsert(self, dataframe, schema_table_name)
     |      Upsert the data from a dataframe into a table (schema.table).
     |
     |      :param dataframe: (DF) The dataframe you would like to upsert
     |      :param schema_table_name: (string) The table in which you would like to upsert the RDD
     |
```
 2) `splicemachine.ml.management`: MLManager interface from Python
  ```
    class MLManager(mlflow.tracking.client.MlflowClient)
     |  A class for managing your MLFlow Runs/Experiments
     |
     |  Method resolution order:
     |      MLManager
     |      mlflow.tracking.client.MlflowClient
     |      builtins.object
     |
     |  Methods defined here:
     |
     |  __init__(self, _tracking_uri='http://mlflow-0-node.hello.mesos:5001')
     |      :param tracking_uri: Address of local or remote tracking server. If not provided, defaults
     |                           to the service set by ``mlflow.tracking.set_tracking_uri``. See
     |                           `Where Runs Get Recorded <../tracking.html#where-runs-get-recorded>`_
     |                           for more info.
     |
     |  __repr__(self)
     |      Return repr(self).
     |
     |  __str__(self)
     |      Return str(self).
     |
     |  create_experiment(self, experiment_name, reset=False)
     |      Create a new experiment. If the experiment
     |      already exists, it will be set to active experiment.
     |      If the experiment doesn't exist, it will be created
     |      and set to active. If the reset option is set to true
     |      (please use with caution), the runs within the existing
     |      experiment will be deleted
     |      :param experiment_name: (str) the name of the experiment to create
     |      :param reset: (bool) whether or not to overwrite the existing run
     |
     |  create_new_run(self, user_id='splice')
     |      Create a new run in the active experiment and set it to be active
     |      :param user_id: the user who creates the run in the MLFlow UI
     |
     |  log_artifact(self, *args, **kwargs)
     |      Log an artifact for the active run
     |
     |  log_artifacts(self, *args, **kwargs)
     |      Log artifacts for the active run
     |
     |  log_metric(self, *args, **kwargs)
     |      Log a metric for the active run
     |
     |  log_model(self, model, module)
     |      Log a model for the active run
     |      :param model: the fitted model/pipeline (in spark) to log
     |      :param module: the module that this is part of (mlflow.spark, mlflow.sklearn etc)
     |
     |  log_param(self, *args, **kwargs)
     |      Log a parameter for the active run
     |
     |  log_spark_model(self, model)
     |      Log a spark model
     |      mlflow.tracking.client.MlflowClient
     |      builtins.object
     |
     |  reset_run(self)
     |      Reset the current run (deletes logged parameters, metrics, artifacts etc.)
     |
     |  set_active_experiment(self, experiment_name)
     |      Set the active experiment of which all new runs will be created under
     |      Does not apply to already created runs
     |
     |      :param experiment_name: either an integer (experiment id) or a string (experiment name)
     |
     |  set_active_run(self, run_id)
     |      Set the active run to a previous run (allows you to log metadata for completed run)
     |      :param run_id: the run UUID for the previous run
     |
     |  set_tag(self, *args, **kwargs)
     |      Set a tag for the active run
     |
     |  delete_experiment(self, experiment_id)
     |      Delete an experiment from the backend store.
     |
     |      :param experiment_id: The experiment ID returned from ``create_experiment``.
     |
     |  delete_run(self, run_id)
     |      Deletes a run with the given ID.
     |
     |  download_artifacts(self, run_id, path)
     |      Download an artifact file or directory from a run to a local directory if applicable,
     |      and return a local path for it.
     |
     |      :param run_id: The run to download artifacts from.
     |      :param path: Relative source path to the desired artifact.
     |      :return: Local path of desired artifact.
     |
     |  get_experiment(self, experiment_id)
     |      :param experiment_id: The experiment ID returned from ``create_experiment``.
     |      :return: :py:class:`mlflow.entities.Experiment`
     |
     |  get_experiment_by_name(self, name)
     |      :param name: The experiment name.
     |      :return: :py:class:`mlflow.entities.Experiment`
     |
     |  get_metric_history(self, run_id, key)
     |      Return a list of metric objects corresponding to all values logged for a given metric.
     |
     |      :param run_id: Unique identifier for run
     |      :param key: Metric name within the run
     |
     |      :return: A list of :py:class:`mlflow.entities.Metric` entities if logged, else empty list
     |
     |  get_run(self, run_id)
     |      Fetch the run from backend store. The resulting :py:class:`Run <mlflow.entities.Run>`
     |      contains a collection of run metadata -- :py:class:`RunInfo <mlflow.entities.RunInfo>`,
     |      as well as a collection of run parameters, tags, and metrics --
     |      :py:class:`RunData <mlflow.entities.RunData>`. In the case where multiple metrics with the
     |      same key are logged for the run, the :py:class:`RunData <mlflow.entities.RunData>` contains
     |      the most recently logged value at the largest step for each metric.
     |
     |      :param run_id: Unique identifier for the run.
     |
     |      :return: A single :py:class:`mlflow.entities.Run` object, if the run exists. Otherwise,
     |               raises an exception.
     |
     |  list_artifacts(self, run_id, path=None)
     |      List the artifacts for a run.
     |
     |      :param run_id: The run to list artifacts from.
     |      :param path: The run's relative artifact path to list from. By default it is set to None
     |                   or the root artifact path.
     |      :return: List of :py:class:`mlflow.entities.FileInfo`
     |
     |  list_experiments(self, view_type=None)
     |      :return: List of :py:class:`mlflow.entities.Experiment`
     |
     |  list_run_infos(self, experiment_id, run_view_type=1)
     |      :return: List of :py:class:`mlflow.entities.RunInfo`
     |
     |  log_batch(self, run_id, metrics, params, tags)
     |      Log multiple metrics, params, and/or tags.
     |
     |      :param metrics: List of Metric(key, value, timestamp) instances.
     |      :param params: List of Param(key, value) instances.
     |      :param tags: List of RunTag(key, value) instances.
     |
     |      Raises an MlflowException if any errors occur.
     |      :returns: None
     |
     |  rename_experiment(self, experiment_id, new_name)
     |      Update an experiment's name. The new name must be unique.
     |
     |      :param experiment_id: The experiment ID returned from ``create_experiment``.
     |
     |  restore_experiment(self, experiment_id)
     |      Restore a deleted experiment unless permanently deleted.
     |
     |      :param experiment_id: The experiment ID returned from ``create_experiment``.
     |
     |  restore_run(self, run_id)
     |      Restores a deleted run with the given ID.
     |
     |  search_runs(self, experiment_ids, filter_string='', run_view_type=1, max_results=1000)
     |      Search experiments that fit the search criteria.
     |
     |      :param experiment_ids: List of experiment IDs, or a single int or string id.
     |      :param filter_string: Filter query string, defaults to searching all runs.
     |      :param run_view_type: one of enum values ACTIVE_ONLY, DELETED_ONLY, or ALL runs
     |                            defined in :py:class:`mlflow.entities.ViewType`.
     |      :param max_results: Maximum number of runs desired.
     |
     |      :return: A list of :py:class:`mlflow.entities.Run` objects that satisfy the search
     |          expressions
     |
     |  set_terminated(self, run_id, status=None, end_time=None)
     |      Set a run's status to terminated.
     |
     |      :param status: A string value of :py:class:`mlflow.entities.RunStatus`.
     |                     Defaults to "FINISHED".
     |      :param end_time: If not provided, defaults to the current time.
```
 3) `splicemachine.ml.utilities`: houses utilities for machine learning
```
    class SpliceBaseEvaluator(builtins.object)
     |  Base ModelEvaluator
     |
     |  Methods defined here:
     |
     |  __init__(self, spark, evaluator, supported_metrics, prediction_column='prediction', label_column='label')
     |      Constructor for SpliceBaseEvaluator
     |      :param spark: spark from zeppelin
     |      :param evaluator: evaluator class from spark
     |      :param supported_metrics: supported metrics list
     |      :param prediction_column: prediction column
     |      :param label_column: label column
     |
     |  get_results(self, dict=False)
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

    class SpliceBinaryClassificationEvaluator(builtins.object)
     |  A Function that provides an easy way to evaluate models once, or over random iterations
     |
     |  Methods defined here:
     |
     |  __init__(self, spark, label_column='label', prediction_column='prediction', confusion_matrix=True)
     |      :param label_column: the column in the dataframe containing the correct output
     |      :param prediction_column: the column in the dataframe containing the prediction
     |      :param confusion_matrix: whether or not to show a confusion matrix after each input
     |
     |  get_results(self, output_type='dataframe')
     |      Return a dictionary containing evaluated results
     |      :param output_type: either a dataframe or a dict (which to return)
     |      :return results: computed_metrics (dict) or computed_df (df)
     |
     |  input(self, predictions_dataframe)
     |      Evaluate actual vs Predicted in a dataframe
     |      :param predictions_dataframe: the dataframe containing the label and the predicition
     |
    class SpliceMultiClassificationEvaluator(SpliceBaseEvaluator)
     |  Base ModelEvaluator
     |
     |  Method resolution order:
     |      SpliceMultiClassificationEvaluator
     |      SpliceBaseEvaluator
     |      builtins.object
     |
     |  Methods defined here:
     |
     |  __init__(self, spark, prediction_column='prediction', label_column='label')
     |      Constructor for SpliceBaseEvaluator
     |      :param spark: spark from zeppelin
     |      :param evaluator: evaluator class from spark
     |      :param supported_metrics: supported metrics list
     |      :param prediction_column: prediction column
     |      :param label_column: label column
     |
     |  ----------------------------------------------------------------------
     |  Methods inherited from SpliceBaseEvaluator:
     |
     |  get_results(self, dict=False)
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
    class SpliceRegressionEvaluator(SpliceBaseEvaluator)
     |  Splice Regression Evaluator
     |
     |  Method resolution order:
     |      SpliceRegressionEvaluator
     |      SpliceBaseEvaluator
     |      builtins.object
     |
     |  Methods defined here:
     |
     |  __init__(self, spark, prediction_column='prediction', label_column='label')
     |      Constructor for SpliceBaseEvaluator
     |      :param spark: spark from zeppelin
     |      :param evaluator: evaluator class from spark
     |      :param supported_metrics: supported metrics list
     |      :param prediction_column: prediction column
     |      :param label_column: label column
     |
     |  ----------------------------------------------------------------------
     |  Methods inherited from SpliceBaseEvaluator:
     |
     |  get_results(self, dict=False)
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
    display(html)
        Display HTML from python in html -- for Zeppelin
        :param html: html string
        :return:
        * note! you may not print out anything in normal python after using this function
        in that cell

    get_confusion_matrix(spark, TP, TN, FP, FN)
        function that shows you a device called a confusion matrix... will be helpful when evaluating.
        It allows you to see how well your model performs
        :param TP: True Positives
        :param TN: True Negatives
        :param FP: False Positives
        :param FN: False Negatives

    print_horizontal_line(l)
        Print a horizontal line l digits long
        :param l: num
        :return: none
```