"""
Copyright 2020 Splice Machine, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.\n

======================================================================================================================================================================================\n

All functions in this module are accessible through the mlflow object and are to be referenced without the leading underscore as \n
.. code-block:: python

    mlflow.function_name()

For example, the function _current_exp_id() is accessible via\n
.. code-block:: python

    mlflow.current_exp_id()


All functions are accessible after running the following import\n
.. code-block:: python

    from splicemachine.mlflow_support import *

Importing anything directly from mlflow before running the above statement will cause problems. After running the above import, you can import additional mlflow submodules as normal\n
.. code-block:: python

    from splicemachine.mlflow_support import *
    from mlflow.tensorflow import autolog

======================================================================================================================================================================================\n
"""
import time
from tempfile import TemporaryDirectory
from collections import defaultdict
from contextlib import contextmanager
import os
from os import path
from sys import version as py_version, stderr
from zipfile import ZipFile, ZIP_DEFLATED
from io import BytesIO

import gorilla
import mlflow
import mlflow.keras
import mlflow.sklearn
import mlflow.spark
import pyspark
import requests
import sklearn
import h2o

from mleap.pyspark import spark_support
from pyspark.ml.base import Model as SparkModel
from pandas.core.frame import DataFrame as PandasDF
from requests.auth import HTTPBasicAuth
from sklearn.base import BaseEstimator as ScikitModel
from tensorflow import __version__ as tf_version
from tensorflow.keras import Model as KerasModel
from tensorflow.keras import __version__ as keras_version
from h2o.estimators.estimator_base import ModelBase as H2OModel

from splicemachine.mlflow_support.constants import FileExtensions, DBLibraries, SparkModelType, H2OModelType, \
    KerasModelType, SklearnModelType

from splicemachine.mlflow_support.utilities import SparkUtils, H2OUtils, KerasUtils, SKUtils, SpliceMachineException, \
    ModelUtils, get_pod_uri, insert_artifact, get_user, create_model_deployment_table, create_parsing_trigger, \
    create_prediction_trigger, create_vti_prediction_trigger, get_feature_columns_and_types, validate_primary_key, \
    get_df_for_mleap, alter_model_table, add_model_to_metadata, drop_tables_on_failure, get_model_library

from splicemachine.spark.constants import CONVERSIONS

from splicemachine.spark.context import PySpliceContext

_TESTING = os.environ.get("TESTING", False)

try:
    _TRACKING_URL = get_pod_uri("mlflow", "5001", _TESTING)
except:
    print("It looks like you're running outside the Splice K8s Cloud Service. "
          "You must run mlflow.set_mlflow_uri(<url>) and pass in the URL to the MLFlow UI", file=stderr)
    _TRACKING_URL = ''

_CLIENT = mlflow.tracking.MlflowClient(tracking_uri=_TRACKING_URL)
mlflow.client = _CLIENT

_GORILLA_SETTINGS = gorilla.Settings(allow_hit=True, store_hit=True)
_PYTHON_VERSION = py_version.split('|')[0].strip()


def _mlflow_patch(name):
    """
    Create a MLFlow Patch that applies the default gorilla settings

    :param name: destination name under mlflow package
    :return: decorator for patched function
    """
    return gorilla.patch(mlflow, name, settings=_GORILLA_SETTINGS)


def _get_current_run_data():
    """
    Get the data associated with the current run.
    As of MLFLow 1.6, it currently does not support getting run info from the mlflow.active_run object, so we need it
    to be retrieved via the tracking client.

    :return: active run data object
    """
    return _CLIENT.get_run(mlflow.active_run().info.run_id).data


@_mlflow_patch('get_run_ids_by_name')
def _get_run_ids_by_name(run_name, experiment_id=None):
    """
    Gets a run id from the run name. If there are multiple runs with the same name, all run IDs are returned

    :param run_name: (str) The name of the run
    :param experiment_id: (int) The experiment to search in. If None, all experiments are searched. [Default None]
    :return: (List[str]) List of run ids
    """
    exps = [experiment_id] if experiment_id else _CLIENT.list_experiments()
    run_ids = []
    for exp in exps:
        for run in _CLIENT.search_runs(exp.experiment_id):
            if run_name == run.data.tags['mlflow.runName']:
                run_ids.append(run.data.tags['Run ID'])
    return run_ids


@_mlflow_patch('register_splice_context')
def _register_splice_context(splice_context):
    """
    Register a Splice Context for Spark/Database operations (artifact storage, for example)

    :param splice_context: (PySpliceContext) splice context to input
    :return: None
    """
    assert isinstance(splice_context, PySpliceContext), "You must pass in a PySpliceContext to this method"
    mlflow._splice_context = splice_context


def _check_for_splice_ctx():
    """
    Check to make sure that the user has registered
    a PySpliceContext with the mlflow object before allowing
    spark operations to take place
    """

    if not hasattr(mlflow, '_splice_context'):
        raise SpliceMachineException(
            "You must run `mlflow.register_splice_context(pysplice_context) before "
            "you can run this mlflow operation!"
        )


@_mlflow_patch('current_run_id')
def _current_run_id():
    """
    Retrieve the current run id

    :return: (str) the current run id
    """
    return mlflow.active_run().info.run_uuid


@_mlflow_patch('current_exp_id')
def _current_exp_id():
    """
    Retrieve the current exp id

    :return: (int) the current experiment id
    """
    return mlflow.active_run().info.experiment_id


@_mlflow_patch('lp')
def _lp(key, value):
    """
    Add a shortcut for logging parameters in MLFlow.

    :param key: (str) key for the parameter
    :param value: (str) value for the parameter
    :return: None
    """
    if len(str(value)) > 250 or len(str(key)) > 250:
        raise SpliceMachineException(f'It seems your parameter input is too long. The max length is 250 characters.'
                                     f'Your key is length {len(str(key))} and your value is length {len(str(value))}.')
    mlflow.log_param(key, value)


@_mlflow_patch('lm')
def _lm(key, value, step=None):
    """
    Add a shortcut for logging metrics in MLFlow.

    :param key: (str) key for the parameter
    :param value: (str or int) value for the parameter
    :param step: (int) A single integer step at which to log the specified Metrics. If unspecified, each metric is logged at step zero.
    """
    if len(str(key)) > 250:
        raise SpliceMachineException(f'It seems your metric key is too long. The max length is 250 characters,'
                                     f'but yours is {len(str(key))}')
    mlflow.log_metric(key, value, step=step)


def __get_serialized_mlmodel(model, conda_env=None):
    """
    Populate the Zip buffer with the serialized MLModel
    :param model: (Model) is the trained Spark/SKlearn/H2O/Keras model
          with the current run
    :param conda_env: [optional] specified conda environment
    """
    zip_buffer = ZipFile(BytesIO(), mode="a", compression=ZIP_DEFLATED, allowZip64=False)

    with TemporaryDirectory() as tempdir:
        if isinstance(model, H2OModel):
            import mlflow.h2o
            mlflow.set_tag('splice.h2o_version', h2o.__version__)
            mlflow.h2o.save_model(model, tempdir, conda_env=conda_env)
            file_ext = FileExtensions.h2o
        elif isinstance(model, SparkModel):
            import mlflow.spark
            mlflow.set_tag('splice.spark_version', pyspark.__version__)
            mlflow.spark.save_model(model, tempdir, conda_env=conda_env)
            file_ext = FileExtensions.spark
        elif isinstance(model, ScikitModel):
            import mlflow.sklearn
            mlflow.set_tag('splice.sklearn_version', sklearn.__version__)
            mlflow.sklearn.save_model(model, tempdir, conda_env=conda_env)
            file_ext = FileExtensions.sklearn
        elif isinstance(model, KerasModel):  # We can't handle keras models with a different backend
            import mlflow.keras
            mlflow.set_tag('splice.keras_version', keras_version)
            mlflow.set_tag('splice.tf_version', tf_version)
            mlflow.keras.save_model(model, tempdir, conda_env=conda_env)
            file_ext = FileExtensions.keras
        else:
            raise SpliceMachineException('Model type not supported for logging.'
                                         'Currently we support logging Spark, H2O, SKLearn and Keras (TF backend) models.'
                                         'You can save your model to disk, zip it and run mlflow.log_artifact to save.')

        for root, dirs, files in os.walk(tempdir.name):
            for file in files:
                zip_buffer.write(path.join(root, file))
        return zip_buffer, file_ext


@_mlflow_patch('log_model')
def _log_model(model, conda_env=None, name='model'):
    """
    Log a trained machine learning model

    :param model: (Model) is the trained Spark/SKlearn/H2O/Keras model
        with the current run
    :param name: (str) the run relative name to store the model under. [Deault 'model']
    """
    _check_for_splice_ctx()

    if _get_current_run_data().tags.get('splice.model_name'):  # this function has already run
        raise SpliceMachineException("Only one model is permitted per run.")

    model_class = str(model.__class__)
    mlflow.set_tag('splice.model_name', name)  # read in backend for deployment
    mlflow.set_tag('splice.model_type', model_class)
    mlflow.set_tag('splice.model_py_version', _PYTHON_VERSION)

    run_id = mlflow.active_run().info.run_uuid

    buffer, file_ext = __get_serialized_mlmodel(model, conda_env=conda_env)
    insert_artifact(splice_context=mlflow._splice_context, name=name, run_uuid=run_id, file_ext=file_ext)


@_mlflow_patch('start_run')
def _start_run(run_id=None, tags=None, experiment_id=None, run_name=None, nested=False):
    """
    Start a new run

    :Example:
        .. code-block:: python
    
            mlflow.start_run(run_name='my_run')\n
            # or\n
            with mlflow.start_run(run_name='my_run'):
                ...


    :param tags: a dictionary containing metadata about the current run. \
        For example: \
            { \
                'team': 'pd', \
                'purpose': 'r&d' \
            }
    :param run_name: (str) an optional name for the run to show up in the MLFlow UI. [Default None]
    :param run_id: (str) if you want to reincarnate an existing run, pass in the run id [Default None]
    :param experiment_id: (int) if you would like to create an experiment/use one for this run [Default None]
    :param nested: (bool) Controls whether run is nested in parent run. True creates a nest run [Default False]
    :return: (ActiveRun) the mlflow active run object
    """
    # Get the current running transaction ID for time travel/data governance
    _check_for_splice_ctx()
    db_connection = mlflow._splice_context.getConnection()
    prepared_statement = db_connection.prepareStatement('CALL SYSCS_UTIL.SYSCS_GET_CURRENT_TRANSACTION()')
    x = prepared_statement.executeQuery()
    x.next()
    timestamp = x.getLong(1)
    prepared_statement.close()

    tags = tags if tags else {}
    tags['mlflow.user'] = get_user()
    tags['DB Transaction ID'] = timestamp

    orig = gorilla.get_original_attribute(mlflow, "start_run")
    active_run = orig(run_id=run_id, experiment_id=experiment_id, run_name=run_name, nested=nested)

    for key in tags:
        mlflow.set_tag(key, tags[key])
    if not run_id:
        mlflow.set_tag('Run ID', mlflow.active_run().info.run_uuid)
    if run_name:
        mlflow.set_tag('mlflow.runName', run_name)

    return active_run


@_mlflow_patch('log_pipeline_stages')
def _log_pipeline_stages(pipeline):
    """
    Log the pipeline stages of a Spark Pipeline as params for the run

    :param pipeline: (PipelineModel) fitted/unitted pipeline
    :return: None
    """
    for stage_number, pipeline_stage in enumerate(SparkUtils.get_stages(pipeline)):
        readable_stage_name = SparkUtils.readable_pipeline_stage(pipeline_stage)
        mlflow.log_param('Stage' + str(stage_number), readable_stage_name)


@_mlflow_patch('log_feature_transformations')
def _log_feature_transformations(unfit_pipeline):
    """
    Log feature transformations for an unfit spark pipeline
    Logs --> feature movement through the pipeline

    :param unfit_pipeline: (PipelineModel) unfit spark pipeline to log
    :return: None
    """
    transformations = defaultdict(lambda: [[], None])  # transformations, outputColumn

    for stage in SparkUtils.get_stages(unfit_pipeline):
        input_cols, output_col = SparkUtils.get_cols(stage, get_input=True), SparkUtils.get_cols(stage, get_input=False)
        if input_cols and output_col:  # make sure it could parse transformer
            for column in input_cols:
                first_column_found = SparkUtils.find_spark_transformer_inputs_by_output(transformations, column)
                if first_column_found:  # column is not original
                    for f in first_column_found:
                        transformations[f][1] = output_col
                        transformations[f][0].append(
                            SparkUtils.readable_pipeline_stage(stage))
                else:
                    transformations[column][1] = output_col
                    transformations[column][0].append(SparkUtils.readable_pipeline_stage(stage))

    for column in transformations:
        param_value = ' -> '.join([column] + transformations[column][0] +
                                  [transformations[column][1]])
        mlflow.log_param('Column- ' + column, param_value)


@_mlflow_patch('log_model_params')
def _log_model_params(pipeline_or_model):
    """
    Log the parameters of a fitted spark model or a model stage of a fitted spark pipeline

    :param pipeline_or_model: fitted spark pipeline/fitted spark model
    """
    model = SparkUtils.get_model_stage(pipeline_or_model)

    mlflow.log_param('model', SparkUtils.readable_pipeline_stage(model))
    if hasattr(model, '_java_obj'):
        verbose_parameters = SparkUtils.parse_string_parameters(model._java_obj.extractParamMap())
    elif hasattr(model, 'getClassifier'):
        verbose_parameters = SparkUtils.parse_string_parameters(
            model.getClassifier()._java_obj.extractParamMap())
    else:
        raise Exception("Could not parse model type: " + str(model))
    for param in verbose_parameters:
        try:
            value = float(verbose_parameters[param])
            mlflow.log_param(param.split('-')[0], value)
        except:
            mlflow.log_param(param.split('-')[0], verbose_parameters[param])


@_mlflow_patch('timer')
@contextmanager
def _timer(timer_name, param=True):
    """
    Context manager for logging

    :Example:
        .. code-block:: python
    
            with mlflow.timer('my_timer'): \n
                ...

    :param timer_name: (str) the name of the timer
    :param param: (bool) whether or not to log the timer as a param (default=True). If false, logs as metric.
    :return: None
    """
    try:
        print(f'Starting Code Block {timer_name}...', end=' ')
        t0 = time.time()
        yield
    finally:
        t1 = time.time() - t0
        # Syntactic Sugar
        (mlflow.log_param if param else mlflow.log_metric)(timer_name, t1)
        print('Done.')
        print(
            f"Code Block {timer_name}:\nRan in {round(t1, 3)} secs\nRan in {round(t1 / 60, 3)} mins"
        )


@_mlflow_patch('download_artifact')
def _download_artifact(name, local_path, run_id=None):
    """
    Download the artifact at the given run id (active default) + name to the local path

    :param name: (str) artifact name to load (with respect to the run)
    :param local_path: (str) local path to download the model to. This path MUST include the file extension
    :param run_id: (str) the run id to download the artifact from. Defaults to active run
    :return: None
    """
    _check_for_splice_ctx()
    file_ext = path.splitext(local_path)[1]

    run_id = run_id or mlflow.active_run().info.run_uuid
    blob_data, f_ext = SparkUtils.retrieve_artifact_stream(mlflow._splice_context, run_id, name)

    if not file_ext:  # If the user didn't provide the file (ie entered . as the local_path), fill it in for them
        local_path += f'/{name}.{f_ext}'

    with open(local_path, 'wb') as artifact_file:
        artifact_file.write(blob_data)


@_mlflow_patch('get_model_name')
def _get_model_name(run_id):
    """
    Gets the model name associated with a run or None

    :param run_id: (str) the run_id that the model is stored under
    :return: (str or None) The model name if it exists
    """
    return _CLIENT.get_run(run_id).data.tags.get('splice.model_name')


@_mlflow_patch('load_model')
def _load_model(run_id=None, name=None):
    """
    Download and deserialize a serialized model

    :param run_id: the id of the run to get a model from
        (the run must have an associated model with it named spark_model)
    :param name: the name of the model in the database
    """
    _check_for_splice_ctx()
    run_id = run_id or mlflow.active_run().info.run_uuid
    name = name or _get_model_name(run_id)
    if not name:
        raise SpliceMachineException(f"Uh Oh! Looks like there isn't a model logged with this run ({run_id})!"
                                     "If there is, pass in the name= parameter to this function")
    model_blob, file_ext = SparkUtils.retrieve_artifact_stream(mlflow._splice_context, run_id, name)

    if file_ext == FileExtensions.spark:
        model = SparkUtils.load_spark_model(mlflow._splice_context, model_blob)
    elif file_ext == FileExtensions.h2o:
        model = H2OUtils.load_h2o_model(model_blob)
    elif file_ext == FileExtensions.sklearn:
        model = SKUtils.load_sklearn_model(model_blob)
    elif file_ext == FileExtensions.keras:
        model = KerasUtils.load_keras_model(model_blob)
    else:
        raise SpliceMachineException(f'Model extension {file_ext} was not a supported model type. '
                                     f'Supported model extensions are {FileExtensions.get_valid()}')

    return model


@_mlflow_patch('log_artifact')
def _log_artifact(file_name, name=None, run_uuid=None):
    """
    Log an artifact for the active run

    :Example:
        .. code-block:: python

            with mlflow.start_run():\n
                mlflow.log_artifact('my_image.png')

    :param file_name: (str) the name of the file name to log
    :param name: (str) the name of the run relative name to store the model under
    :param run_uuid: (str) the run uuid of a previous run, if none, defaults to current run
    :return: None
    
    :NOTE: 
        We do not currently support logging directories. If you would like to log a directory, please zip it first and log the zip file
    """
    _check_for_splice_ctx()
    file_ext = path.splitext(file_name)[1].lstrip('.')

    with open(file_name, 'rb') as artifact:
        byte_stream = bytearray(bytes(artifact.read()))

    run_id = run_uuid or mlflow.active_run().info.run_uuid
    name = name or file_name
    insert_artifact(mlflow._splice_context, name, byte_stream, run_id, file_ext=file_ext)


@_mlflow_patch('login_director')
def _login_director(username, password):
    """
    Authenticate into the MLManager Director

    :param username: (str) database username
    :param password: (str) database password
    """
    mlflow._basic_auth = HTTPBasicAuth(username, password)


def _initiate_job(payload, endpoint):
    """
    Send a job to the initiation endpoint

    :param payload: (dict) JSON payload for POST request
    :param endpoint: (str) REST endpoint to target
    :return: (str) Response text from request
    """
    if not hasattr(mlflow, '_basic_auth'):
        raise Exception(
            "You have not logged into MLManager director."
            " Please run mlflow.login_director(username, password)"
        )
    request = requests.post(
        get_pod_uri('mlflow', 5003, _testing=_TESTING) + endpoint,
        auth=mlflow._basic_auth,
        json=payload,

    )

    if request.ok:
        print("Your Job has been submitted. View its status on port 5003 (Job Dashboard)")
        print(request.json)
        return request.json
    else:
        print("Error! An error occurred while submitting your job")
        print(request.text)
        return request.text


@_mlflow_patch('deploy_aws')
def _deploy_aws(app_name, region='us-east-2', instance_type='ml.m5.xlarge',
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
    """
    # get run from mlflow
    print("Processing...")
    time.sleep(3)  # give the mlflow server time to register the artifact, if necessary

    supported_aws_regions = ['us-east-2', 'us-west-1', 'us-west-2', 'eu-central-1']
    supported_instance_types = ['ml.m5.xlarge']
    supported_deployment_modes = ['replace', 'add']

    # data validation
    if region not in supported_aws_regions:
        raise Exception("Region must be in list: " + str(supported_aws_regions))
    if instance_type not in supported_instance_types:
        raise Exception("Instance type must be in list: " + str(instance_type))
    if deployment_mode not in supported_deployment_modes:
        raise Exception("Deployment mode must be in list: " + str(supported_deployment_modes))

    request_payload = {
        'handler_name': 'DEPLOY_AWS', 'run_id': run_id if run_id else mlflow.active_run().info.run_uuid,
        'region': region, 'user': get_user(),
        'instance_type': instance_type, 'instance_count': instance_count,
        'deployment_mode': deployment_mode, 'app_name': app_name
    }

    return _initiate_job(request_payload, '/api/rest/initiate')


@_mlflow_patch('deploy_azure')
def _deploy_azure(endpoint_name, resource_group, workspace, run_id=None, region='East US',
                  cpu_cores=0.1, allocated_ram=0.5, model_name=None):
    """
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
    """
    supported_regions = ['East US', 'East US 2', 'Central US',
                         'West US 2', 'North Europe', 'West Europe', 'Japan East']

    if region not in supported_regions:
        raise Exception("Region must be in list: " + str(supported_regions))
    if cpu_cores <= 0:
        raise Exception("Invalid CPU Count")
    if allocated_ram <= 0:
        raise Exception("Invalid Allocated RAM")

    request_payload = {
        'handler_name': 'DEPLOY_AZURE',
        'endpoint_name': endpoint_name,
        'resource_group': resource_group,
        'workspace': workspace,
        'run_id': run_id if run_id else mlflow.active_run().info.run_uuid,
        'cpu_cores': cpu_cores,
        'allocated_ram': allocated_ram,
        'model_name': model_name
    }
    return _initiate_job(request_payload, '/api/rest/initiate')


@_mlflow_patch('deploy_database')
def _deploy_db(db_schema_name,
               db_table_name,
               run_id,
               primary_key=None,
               df=None,
               create_model_table=False,
               model_cols=None,
               classes=None,
               sklearn_args={},
               verbose=False,
               pred_threshold=None,
               replace=False) -> None:
    """
    Deploy a trained (currently Spark, Sklearn, Keras or H2O) model to the Database.
    This either creates a new table or alters an existing table in the database (depending on parameters passed)

    :param db_schema_name: (str) the schema name to deploy to.
    :param db_table_name: (str) the table name to deploy to.
    :param run_id: (str) The run_id to deploy the model on. The model associated with this run will be deployed
    :param primary_key: (List[Tuple[str, str]]) List of column + SQL datatype to use for the primary/composite key. \n
        * If you are deploying to a table that already exists, it must already have a primary key, and this parameter will be ignored. \n
        * If you are creating the table in this function, you MUST pass in a primary key 
    :param df: (Spark or Pandas DF) The dataframe used to train the model \n
                | NOTE: The columns in this df are the ones that will be used to create the table unless specified by model_cols
    :param create_model_table: Whether or not to create the table from the dataframe. Default false. This
                                Will ONLY be used if the table does not exist and a dataframe is passed in
    :param model_cols: (List[str]) The columns from the table to use for the model. If None, all columns in the table
                                        will be passed to the model. If specified, the columns will be passed to the model
                                        IN THAT ORDER. The columns passed here must exist in the table.
    :param classes: (List[str]) The classes (prediction labels) for the model being deployed.\n
                    NOTE: If not supplied, the table will have default column names for each class
    :param sklearn_args: (dict{str: str}) Prediction options for sklearn models: \n
        * Available key value options: \n
            * 'predict_call': 'predict', 'predict_proba', or 'transform' \n
                * Determines the function call for the model \n
                        * If blank, predict will be used (or transform if model doesn't have predict) \n
            * 'predict_args': 'return_std' or 'return_cov' - For Bayesian and Gaussian models \n
                * Only one can be specified \n
                    * If the model does not have the option specified, it will be ignored.
    :param verbose: (bool) Whether or not to print out the queries being created. Helpful for debugging
    :param pred_threshold: (double) A prediction threshold for *Keras* binary classification models \n
        * If the model type isn't Keras, this parameter will be ignored \n
        NOTE: If the model type is Keras, the output layer has 1 node, and pred_threshold is None, \
                you will NOT receive a class prediction, only the output of the final layer (like model.predict()). \
                If you want a class prediction \
                for your binary classification problem, you MUST pass in a threshold.
    :param replace: (bool) whether or not to replace a currently existing model. This param does not yet work
    :return: None\n
    
    This function creates the following IF you are creating a table from the dataframe \n
        * The model table where run_id is the run_id passed in. This table will have a column for each feature in the feature vector. It will also contain:\n
            * USER which is the current user who made the request
            * EVAL_TIME which is the CURRENT_TIMESTAMP
            * the PRIMARY KEY column(s) passed in
            * PREDICTION. The prediction of the model. If the :classes: param is not filled in, this will be default values for classification models
            * A column for each class of the predictor with the value being the probability/confidence of the model if applicable\n
    IF you are deploying to an existing table, the table will be altered to include the columns above. \n
    :NOTE:
        .. code-block:: text
    
            The columns listed above are default value columns.\n 
            This means that on a SQL insert into the table, \n
            you do not need to reference or insert values into them.\n
            They are automatically taken care of.\n
            Set verbose=True in the function call for more information

    The following will also be created for all deployments: \n
        * A trigger that runs on (after) insertion to the data table that runs an INSERT into the prediction table, \
            calling the PREDICT function, passing in the row of data as well as the schema of the dataset, and the run_id of the model to run \n
        * A trigger that runs on (after) insertion to the prediction table that calls an UPDATE to the row inserted, \
            parsing the prediction probabilities and filling in proper column values
    """
    _check_for_splice_ctx()

    # Get the model
    run_id = run_id if run_id else mlflow.active_run().info.run_uuid
    fitted_model = _load_model(run_id)

    # Param checking. Can't create model table without a dataframe
    if create_model_table and df is None:  # Need to compare to None, truth value of df is ambiguous
        raise SpliceMachineException(
            "If you'd like to create the model table as part of this deployment, you must pass in a dataframe")
    # Make sure primary_key is valid format
    if create_model_table and not primary_key:
        raise SpliceMachineException(
            "If you'd like to create the model table as part of this deployment must provide the primary key(s)")

    # FIXME: We need to use the dbConnection so we can set a savepoint and rollback on failure
    classes = classes if classes else []

    schema_table_name = f'{db_schema_name}.{db_table_name}'

    feature_columns, schema_types = get_feature_columns_and_types(mlflow._splice_context, df, create_model_table,
                                                                  model_cols, schema_table_name)

    # Validate primary key is correct, or that provided table has primary keys
    primary_key = validate_primary_key(mlflow._splice_context, primary_key, db_schema_name,
                                       db_table_name) or primary_key

    library = get_model_library(fitted_model)
    if library == DBLibraries.MLeap:
        # Mleap needs a dataframe in order to serialize the model
        df = get_df_for_mleap(mlflow._splice_context, schema_table_name, df)

    model_type, classes, model_already_exists = ModelUtils[library].prep_model_for_deployment(mlflow._splice_context,
                                                                                              fitted_model, classes,
                                                                                              run_id,
                                                                                              df, pred_threshold,
                                                                                              sklearn_args)

    print(f'Deploying model {run_id} to table {schema_table_name}')

    # Create the schema of the table (we use this a few times)
    schema_str = ''
    for i in feature_columns:
        schema_str += f'\t{i} {CONVERSIONS[schema_types[str(i)]]},'

    try:
        # Create/Alter table 1: DATA
        if create_model_table:
            print('Creating model table ...', end=' ')
            create_model_deployment_table(mlflow._splice_context, run_id, schema_table_name, schema_str, classes,
                                          primary_key, model_type, verbose)
            print('Done.')
        else:
            print('Altering provided table for deployment')
            alter_model_table(mlflow._splice_context, run_id, schema_table_name, classes, model_type, verbose)

        # Create Trigger 1: model prediction
        print('Creating model prediction trigger ...', end=' ')
        if model_type in (H2OModelType.KEY_VALUE, SklearnModelType.KEY_VALUE, KerasModelType.KEY_VALUE):
            create_vti_prediction_trigger(mlflow._splice_context, schema_table_name, run_id, feature_columns,
                                          schema_types,
                                          schema_str, primary_key, classes, model_type, sklearn_args, pred_threshold,
                                          verbose)
        else:
            create_prediction_trigger(mlflow._splice_context, schema_table_name, run_id, feature_columns, schema_types,
                                      schema_str, primary_key, model_type, verbose)
        print('Done.')

        if model_type in (SparkModelType.CLASSIFICATION, SparkModelType.CLUSTERING_WITH_PROB,
                          H2OModelType.CLASSIFICATION):
            # Create Trigger 2: model parsing
            print('Creating parsing trigger ...', end=' ')
            create_parsing_trigger(mlflow._splice_context, schema_table_name, primary_key, run_id, classes, model_type,
                                   verbose)
            print('Done.')

        add_model_to_metadata(mlflow._splice_context, run_id, schema_table_name)


    except Exception as e:
        import traceback
        exc = 'Model deployment failed. Rolling back transactions.\n'
        print(exc)
        drop_tables_on_failure(mlflow._splice_context, schema_table_name, run_id, model_already_exists)
        if not verbose:
            exc += 'For more insight into the SQL statement that generated this error, rerun with verbose=True'
        traceback.print_exc()
        raise SpliceMachineException(exc)

    print('Model Deployed.')


@_mlflow_patch('get_deployed_models')
def _get_deployed_models() -> PandasDF:
    """
    Get the currently deployed models in the database
    :return: Pandas df
    """

    return mlflow._splice_context.df(
        """
        SELECT * FROM MLMANAGER.LIVE_MODEL_STATUS
        """
    ).toPandas()


def apply_patches():
    """
    Apply all the Gorilla Patches; \
    All Gorilla Patched MUST be predixed with '_' before their destination in MLflow
    """
    targets = [_register_splice_context, _lp, _lm, _timer, _log_artifact, _log_feature_transformations,
               _log_model_params, _log_pipeline_stages, _log_model, _load_model, _download_artifact,
               _start_run, _current_run_id, _current_exp_id, _deploy_aws, _deploy_azure, _deploy_db, _login_director,
               _get_run_ids_by_name, _get_deployed_models]

    for target in targets:
        gorilla.apply(gorilla.Patch(mlflow, target.__name__.lstrip('_'), target, settings=_GORILLA_SETTINGS))


def set_mlflow_uri(uri):
    """
    Set the tracking uri for mlflow. Only needed if running outside of the Splice Machine K8s Cloud Service

    :param uri: (str) the URL of your mlflow UI.
    :return: None
    """
    _CLIENT = uri
    mlflow.client = _CLIENT
    mlflow.set_tracking_uri(uri)


def main():
    mlflow.set_tracking_uri(_TRACKING_URL)
    apply_patches()


main()
