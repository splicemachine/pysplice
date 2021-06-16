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
import glob
import os
import time
import copy
from collections import defaultdict
from contextlib import contextmanager
from importlib import import_module
from io import BytesIO
from os import path
from sys import version as py_version, stderr
from tempfile import TemporaryDirectory, NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile
from typing import Dict, Optional, List, Union

import gorilla
import h2o
import mlflow
import mlflow.pyfunc
from mlflow.tracking.fluent import ActiveRun
from mlflow.entities import RunStatus
import pyspark
import requests
import sklearn
import yaml
import warnings

from pandas.core.frame import DataFrame as PandasDF
from pyspark.ml.base import Model as SparkModel
from pyspark.sql import DataFrame as SparkDF
from requests.auth import HTTPBasicAuth
from sklearn.base import BaseEstimator as ScikitModel

from splicemachine.features import FeatureStore
from splicemachine.mlflow_support.constants import (FileExtensions, DatabaseSupportedLibs)
from splicemachine.mlflow_support.utilities import (SparkUtils, get_pod_uri, get_user,
                                                    insert_artifact, download_artifact, get_jobs_uri)
from splicemachine import SpliceMachineException
from splicemachine.spark.context import PySpliceContext

try: # PySpark/H2O 3.X
    from h2o.model.model_base import ModelBase as H2OModel
except: # PySpark/H2O 2.X
    from h2o.estimators.estimator_base import ModelBase as H2OModel

# For recording notebook history
try:
    from IPython import get_ipython
    import nbformat as nbf
    ipython = get_ipython()
    mlflow._notebook_history = bool(ipython) # If running outside a notebook/ipython, this will be False
except:
    mlflow._notebook_history = False

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

class SpliceActiveRun(ActiveRun):
    """
    A wrapped active run for Splice Machine that calls our custom mlflow.end_run, so we can record the notebook
    history
    """
    def __exit__(self, exc_type, exc_val, exc_tb):
        status = RunStatus.FINISHED if exc_type is None else RunStatus.FAILED
        mlflow.end_run(RunStatus.to_string(status))
        return exc_type is None

def __try_auto_login():
    """
    Tries to login the user to the Director for deployment automatically. This will only work if the user is not
    using the cloud service.

    :return: None
    """
    jwt = os.environ.get('SPLICE_JUPYTER_JWTTOKEN')
    if jwt:
        mlflow.login_director(jwt_token=jwt)
    user, password = os.environ.get('SPLICE_JUPYTER_USER'), os.environ.get('SPLICE_JUPYTER_PASSWORD')
    if user and password:
        mlflow.login_director(username=user, password=password)

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
    return mlflow.client.get_run(mlflow.active_run().info.run_id).data

def __get_active_user():
    if hasattr(mlflow, '_username'):
        return mlflow._username
    if get_user():
        return get_user()
    return SpliceMachineException("Could not detect active user. Please run mlflow.login_director() and pass in your Splice"
                                  "username and password or JWT token.")

@_mlflow_patch('get_run_ids_by_name')
def _get_run_ids_by_name(run_name, experiment_id=None):
    """
    Gets a run id from the run name. If there are multiple runs with the same name, all run IDs are returned

    :param run_name: (str) The name of the run
    :param experiment_id: (int) The experiment to search in. If None, all experiments are searched. [Default None]
    :return: (List[str]) List of run ids
    """
    exps = [mlflow.client.get_experiment(experiment_id)] if experiment_id else mlflow.client.list_experiments()
    run_ids = []
    for exp in exps:
        for run in mlflow.client.search_runs(exp.experiment_id):
            if run_name == run.data.tags.get('mlflow.runName'):
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

@_mlflow_patch('register_feature_store')
def _register_feature_store(fs: FeatureStore):
    """
    Register a feature store for feature tracking of experiments

    :param feature_store: (FeatureStore) The feature store
    :return: None
    """
    mlflow._feature_store = fs
    mlflow._feature_store.mlflow_ctx = mlflow

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


def __get_serialized_mlmodel(model, model_lib=None, **flavor_options):
    """
    Populate the Zip buffer with the serialized MLModel
    :param model: (Model) is the trained Spark/SKlearn/H2O/Keras model
          with the current run
    :param flavor_options: The extra kw arguments to any particular model library. If this is set, model_lib must be set
    """
    buffer = BytesIO()
    zip_buffer = ZipFile(buffer, mode="a", compression=ZIP_DEFLATED, allowZip64=False)
    with TemporaryDirectory() as tempdir:
        mlmodel_dir = f'{tempdir}/model'
        if model_lib:
            try:
                import mlflow
                import_module(f'mlflow.{model_lib}')
                if model_lib == 'pyfunc':
                    getattr(mlflow, model_lib).save_model(python_model=model, path=mlmodel_dir, **flavor_options)
                else:
                    getattr(mlflow, model_lib).save_model(model, path=mlmodel_dir, **flavor_options)

                file_ext = FileExtensions.map_from_mlflow_flavor(model_lib) if \
                    model_lib in DatabaseSupportedLibs.get_valid() else model_lib

            except Exception as e:
                print(str(e))
                raise SpliceMachineException(f'Failed to save model type {model_lib}. Ensure that is a supposed model '
                                             f'flavor https://www.mlflow.org/docs/1.8.0/models.html#built-in-model-flavors\n'
                                             f'Or you can build a pyfunc model\n'
                                             'https://www.mlflow.org/docs/1.8.0/models.html#python-function-python-function')
        # deprecated behavior
        elif isinstance(model, H2OModel):
            import mlflow.h2o
            mlflow.set_tag('splice.h2o_version', h2o.__version__)
            mlflow.h2o.save_model(model, mlmodel_dir, **flavor_options)
            file_ext = FileExtensions.h2o
        elif isinstance(model, SparkModel):
            import mlflow.spark
            mlflow.set_tag('splice.spark_version', pyspark.__version__)
            mlflow.spark.save_model(model, mlmodel_dir, **flavor_options)
            file_ext = FileExtensions.spark
        elif isinstance(model, ScikitModel):
            import mlflow.sklearn
            mlflow.set_tag('splice.sklearn_version', sklearn.__version__)
            mlflow.sklearn.save_model(model, mlmodel_dir, **flavor_options)
            file_ext = FileExtensions.sklearn
        else:
            raise SpliceMachineException('Model type not supported for logging. If you received this error,'
                                         'you should pass a value to the model_lib parameter of the model type you '
                                         'want to save, or call the original mlflow.<flavor>.log_model(). '
                                         'Supported values are available here: '
                                         'https://www.mlflow.org/docs/1.8.0/models.html#built-in-model-flavors\n'
                                         'as well as \'pyfunc\' '
                                         'https://www.mlflow.org/docs/1.8.0/models.html#python-function-python-function')

        for model_file in glob.glob(mlmodel_dir + "/**/*", recursive=True):
            zip_buffer.write(model_file, arcname=path.relpath(model_file, mlmodel_dir))

        return buffer, file_ext


@_mlflow_patch('end_run')
def _end_run(status=RunStatus.to_string(RunStatus.FINISHED), save_html=True):
    """End an active MLflow run (if there is one).

    .. code-block:: python
        :caption: Example

        import mlflow

        # Start run and get status
        mlflow.start_run()
        run = mlflow.active_run()
        print("run_id: {}; status: {}".format(run.info.run_id, run.info.status))

        # End run and get status
        mlflow.end_run()
        run = mlflow.get_run(run.info.run_id)
        print("run_id: {}; status: {}".format(run.info.run_id, run.info.status))
        print("--")

        # Check for any active runs
        print("Active run: {}".format(mlflow.active_run()))

    .. code-block:: text
        :caption: Output

        run_id: b47ee4563368419880b44ad8535f6371; status: RUNNING
        run_id: b47ee4563368419880b44ad8535f6371; status: FINISHED
        --
        Active run: None
    """
    if mlflow._notebook_history and mlflow.active_run():
        try:
            with NamedTemporaryFile() as temp_file:
                nb = nbf.v4.new_notebook()
                nb['cells'] = [nbf.v4.new_code_cell(code) for code in ipython.history_manager.input_hist_raw]
                nbf.write(nb, temp_file.name)
                run_name = mlflow.get_run(mlflow.current_run_id()).to_dictionary()['data']['tags']['mlflow.runName']
                mlflow.log_artifact(temp_file.name, name=f'{run_name}_run_log.ipynb')
                typ,ext = ('html','html') if save_html else ('script','py')
                os.system(f'jupyter nbconvert --to {typ} {temp_file.name}')
                mlflow.log_artifact(f'{temp_file.name[:-1]}.{ext}', name=f'{run_name}_run_log.{ext}')
                os.system(f'pip freeze > {temp_file.name}.txt')
                mlflow.log_artifact(f'{temp_file.name}.txt', name='pip_env.txt')

        except:
            warnings.warn('There was an issue storing the run log history for this mlflow run. Your run will not have '
                          'the run log.')
    orig = gorilla.get_original_attribute(mlflow, "end_run")
    orig(status=status)

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
    tags = tags or {}
    tags['mlflow.user'] = __get_active_user()

    orig = gorilla.get_original_attribute(mlflow, "start_run")
    active_run = orig(run_id=run_id, experiment_id=experiment_id, run_name=run_name, nested=nested)

    for key in tags:
        mlflow.set_tag(key, tags[key])
    if not run_id:
        mlflow.set_tag('Run ID', mlflow.active_run().info.run_uuid)
    if run_name:
        mlflow.set_tag('mlflow.runName', run_name)
    if hasattr(mlflow,'_active_training_set'):
        mlflow._active_training_set._register_metadata(mlflow)

    return SpliceActiveRun(active_run)

@_mlflow_patch('remove_active_training_set')
def _remove_active_training_set():
    """
    Removes the active training set from mlflow. This function deletes mlflows active training set (retrieved from
    the feature store), which will in turn stop the automated logging of features to the active mlflow run. To recreate
    an active training set, call fs.get_training_set or fs.get_training_set_from_view in the Feature Store.
    """
    if hasattr(mlflow,'_active_training_set'):
        del mlflow._active_training_set


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
        value = verbose_parameters[param]
        if value: # Spark 3.0 leafCol returns an empty parameter, mlflow fails if you try to log an empty string
            try:
                value = float(value)
                mlflow.log_param(param.split('-')[0], value)
            except:
                mlflow.log_param(param.split('-')[0], value)


@_mlflow_patch('timer')
@contextmanager
def _timer(timer_name, param=False):
    """
    Context manager for logging

    :Example:
        .. code-block:: python
    
            with mlflow.timer('my_timer'): \n
                ...

    :param timer_name: (str) the name of the timer
    :param param: (bool) whether or not to log the timer as a param (default=True). If false, logs as metric.
    """
    t0 = time.time()
    try:
        print(f'Starting Code Block {timer_name}...', end=' ')
        yield
    finally:
        t1 = time.time() - t0
        # Syntactic Sugar
        (mlflow.log_param if param else mlflow.log_metric)(timer_name, t1)
        print('Done.')
        print(
            f"Code Block {timer_name}:\nRan in {round(t1, 3)} secs\nRan in {round(t1 / 60, 3)} mins"
        )


@_mlflow_patch('get_model_name')
def _get_model_name(run_id):
    """
    Gets the model name associated with a run or None

    :param run_id: (str) the run_id that the model is stored under
    :return: (str or None) The model name if it exists
    """
    return mlflow.client.get_run(run_id).data.tags.get('splice.model_name')

@_mlflow_patch('log_model')
def _log_model(model, name='model', model_lib=None, **flavor_options):
    """
    Log a trained machine learning model

    :param model: (Model) is the trained Spark/SKlearn/H2O/Keras model
        with the current run
    :param name: (str) the run relative name to store the model under. [Deault 'model']
    :param model_lib: An optional param specifying the model type of the model to log
        Available options match the mlflow built-in model flavors https://www.mlflow.org/docs/1.8.0/models.html#built-in-model-flavors
    :param flavor_options: (**kwargs) The full set of save options to pass into the save_model function. If this is passed,
        model_class must also be provided and the keys of this dictionary must match the params of that functions signature
        (ie mlflow.pyfunc.save_model). An example of pyfuncs signature is here, although each flavor has its own.
        https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.save_model
    """
    # Make sure no models have been logged to this run
    if _get_current_run_data().tags.get('splice.model_name'):  # this function has already run
        raise SpliceMachineException("Only one model is permitted per run.")
    if flavor_options and not model_lib:
        raise SpliceMachineException("You cannot set mlflow-flavor specific options without setting the model library. "
                                     "Either set model_lib, or use the native mlflow.<flavor>.log_model function")

    model_class = str(model.__class__)

    if not mlflow.active_run():
        raise SpliceMachineException("You must have an active run to log a model")

    run_id = mlflow.active_run().info.run_uuid
    buffer, file_ext = __get_serialized_mlmodel(model, model_lib=model_lib, **flavor_options)
    buffer.seek(0)
    model_data = buffer.read()

    with NamedTemporaryFile(mode='wb', suffix='.zip') as f:
        f.write(model_data)
        f.seek(0)
        host = get_jobs_uri(mlflow.get_tracking_uri() or get_pod_uri('mlflow', 5003, _testing=_TESTING))
        insert_artifact(host, f.name, name, run_id, file_ext, mlflow._basic_auth, artifact_path=name)

    # Set the model metadata as tags after successful logging
    mlflow.set_tag('splice.model_name', name)  # read in backend for deployment
    mlflow.set_tag('splice.model_type', model_class)
    mlflow.set_tag('splice.model_py_version', _PYTHON_VERSION)


def _load_model(run_id=None, name=None, as_pyfunc=False):
    """
    Download and deserialize a serialized model

    :param run_id: (str) the id of the run to get a model from
        (the run must have an associated model with it named spark_model)
    :param name: (str) the name of the model in the database
    :param as_pyfunc: (bool) load as a model-agnostic pyfunc model
        (https://www.mlflow.org/docs/latest/models.html#python-function-python-function)
    """
    if not (run_id or mlflow.active_run()):
        raise SpliceMachineException("You need to pass in a run_id or start an mlflow run.")

    run_id = run_id or mlflow.active_run().info.run_uuid
    name = name or _get_model_name(run_id)
    if not name:
        raise SpliceMachineException(f"Uh Oh! Looks like there isn't a model logged with this run ({run_id})!"
                                     "If there is, pass in the name= parameter to this function")

    host = get_jobs_uri(mlflow.get_tracking_uri() or get_pod_uri('mlflow', 5003, _testing=_TESTING))
    r = download_artifact(host, name, run_id, mlflow._basic_auth)
    buffer = BytesIO()
    buffer.seek(0)
    buffer.write(r.content)

    with TemporaryDirectory() as tempdir:
        ZipFile(buffer).extractall(path=tempdir)
        if as_pyfunc:
            mlflow_module = 'pyfunc'
        else:
            with open(f'{tempdir}/MLmodel', 'r') as mlmodel_file:
                mlmodel = yaml.safe_load(mlmodel_file.read())
                try:
                    loader_module = mlmodel['flavors']['python_function']['loader_module']
                except KeyError: # If the python_function isn't available, fallback and try the raw model flavor
                    # We will look through the other flavors in the MLModel yaml
                    loader_module = None
                    for flavor in mlmodel['flavors'].keys():
                        if hasattr(mlflow, flavor):
                            loader_module = f'mlflow.{flavor}'
                            break
                    if not loader_module:
                        raise SpliceMachineException(f"Unable to load the mlflow loader. Ensure this ML model has "
                                                 f"been saved using an mlflow module")
            mlflow_module = loader_module.split('.')[1]  # get the mlflow.(MODULE)
            import_module(loader_module)
        return getattr(mlflow, mlflow_module).load_model(tempdir)


@_mlflow_patch('log_artifact')
def _log_artifact(file_name, name=None, run_uuid=None, artifact_path = None):
    """
    Log an artifact for the active run

    :Example:
        .. code-block:: python

            with mlflow.start_run():\n
                mlflow.log_artifact('my_image.png')

    :param file_name: (str) the name of the file name to log
    :param name: (str) the name to store the artifact as. Defaults to the file name. If the name param includes the file
        extension (or is not passed in) you will be able to preview it in the mlflow UI (image, text, html, geojson files).
    :param run_uuid: (str) the run uuid of a previous run, if none, defaults to current run
    :param artifact_path: If you would like the artifact logged as a subdirectory of an particular folder,
        you can set this value. If the directory doesn't exist, it will be created for this run's artifact path.
    :return: None
    
    :NOTE: 
        We do not currently support logging directories. If you would like to log a directory, please zip it first and log the zip file
    """
    if not os.path.exists(file_name):
        raise SpliceMachineException(f'Cannot find file {file_name}')
    # Check file size without reading file
    if os.path.getsize(file_name) > 5e7:
        raise SpliceMachineException(f'File {file_name} is too large. Max file size is 50MB')

    file_ext = path.splitext(file_name)[1].lstrip('.')

    run_id = run_uuid or mlflow.active_run().info.run_uuid
    name = name or os.path.split(file_name)[-1]

    host = get_jobs_uri(mlflow.get_tracking_uri() or get_pod_uri('mlflow', 5003, _testing=_TESTING))
    insert_artifact(host, file_name, name, run_id, file_ext, mlflow._basic_auth, artifact_path)
    print(f'Saved artifact as {name} in mlflow')

@_mlflow_patch('download_artifact')
def _download_artifact(name, local_path=None, run_id=None):
    """
    Download the artifact at the given run id (active default) + name to the local path

    :param name: (str) artifact name to load (with respect to the run)
    :param local_path: (str) local path to download the model to. If set, this path MUST include the file extension.
        Will default to the current directory and the name of the saved artifact
    :param run_id: (str) the run id to download the artifact from. Defaults to active run
    """
    run_id = run_id or mlflow.active_run().info.run_uuid
    host = get_jobs_uri(mlflow.get_tracking_uri() or get_pod_uri('mlflow', 5003, _testing=_TESTING))
    r = download_artifact(host, name, run_id, mlflow._basic_auth)
    file_name = local_path or r.headers['Content-Disposition'].split('filename=')[1]
    with open(file_name, 'wb') as file:
        file.write(r.content)
    print(f'Done. File has been written to {file_name}')

@_mlflow_patch('login_director')
def _login_director(username=None, password=None, jwt_token=None):
    """
    Authenticate into the MLManager Director

    :param username: (str) database username
    :param password: (str) database password
    :param jwt_token: (str) database JWT token authentication

    Either (username/password) for basic auth or jwt_token must be provided. Basic authentication takes precedence if set (mlflow default)
    """
    if (username and not password) or (password and not username):
        raise SpliceMachineException("You must either set both username and password, or neither. You cannot set just one")

    if username and password:
        mlflow._basic_auth = HTTPBasicAuth(username, password)
        mlflow._username = username
        os.environ['MLFLOW_TRACKING_USERNAME'] = username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = password
    if jwt_token:
        os.environ['MLFLOW_TRACKING_TOKEN'] = jwt_token


def __initiate_job(payload, endpoint):
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
        get_jobs_uri(mlflow.get_tracking_uri() or get_pod_uri('mlflow', 5003, _testing=_TESTING)) + endpoint,
        auth=mlflow._basic_auth,
        json=payload
    )

    if request.ok:
        print("Your Job has been submitted. The returned value of this function is"
              " the job id, which you can use to monitor the your task in real-time. Run mlflow.watch_job(<job id>) to"
              "stream them to stdout, or mlflow.fetch_logs(<job id>) to read them one time to a list")
        return request.json()['job_id']
    else:
        print("Error! An error occurred while submitting your job")
        print(request.text)
        return request.text


@_mlflow_patch('deploy_aws')
def _deploy_aws(app_name: str, region: str = 'us-east-2', instance_type: str = 'ml.m5.xlarge',
                run_id: str = None, instance_count: int = 1, deployment_mode: str = 'replace'):
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

    request_payload = {
        'handler_name': 'DEPLOY_AWS', 'run_id': run_id,
        'region': region, 'user': __get_active_user(),
        'instance_type': instance_type, 'instance_count': instance_count,
        'deployment_mode': deployment_mode, 'app_name': app_name
    }

    return __initiate_job(request_payload, '/api/rest/initiate')


@_mlflow_patch('deploy_azure')
def _deploy_azure(endpoint_name: str, resource_group: str, workspace: str, run_id: str, region: str = 'East US',
                  cpu_cores: float = 0.1, allocated_ram: float = 0.5, model_name: str = None):
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
    request_payload = {
        'handler_name': 'DEPLOY_AZURE',
        'endpoint_name': endpoint_name,
        'resource_group': resource_group,
        'workspace': workspace,
        'region': region,
        'run_id': run_id,
        'cpu_cores': cpu_cores,
        'allocated_ram': allocated_ram,
        'model_name': model_name
    }
    return __initiate_job(request_payload, '/api/rest/initiate')


@_mlflow_patch('deploy_kubernetes')
def _deploy_kubernetes(run_id: str, service_port: int = 80,
                       base_replicas: int = 1, autoscaling_enabled: bool = False,
                       max_replicas: int = 2, target_cpu_utilization: int = 50,
                       disable_nginx: bool = False, gunicorn_workers: int = 1,
                       resource_requests_enabled: bool = False, resource_limits_enabled: bool = False,
                       cpu_request: int = 0.5, cpu_limit: int = 1, memory_request: str = "512Mi",
                       memory_limit: str = "2048Mi", expose_external: bool = False):
    """
    Deploy model associated with the specified or active run to Kubernetes cluster.\n

    Creates the Following Resources:
        * Pod (with your model loaded in via an init container)
        * ReplicaSet (configured to base replicas specified)
        * HPA (if autoscaling is enabled)
        * Service (model-<run id>.<db namespace>.svc.cluster.local:<service port specified>)
        * Deployment
        * Ingress (if expose enable is set to True) (on <your cluster url>/<run id>/invocations)

    :param run_id: specified if overriding the active run
    :param service_port: (default 80) the port that the prediction service runs on internally in the cluster
    :param autoscaling_enabled: (default False) whether or not to provision a Horizontal Pod Autoscaler to provision
            pods dynamically
    :param max_replicas: (default 2) [USED IF AUTOSCALING ENABLED] max number of pods to scale up to
    :param target_cpu_utilization: (default 50) [USED IF AUTOSCALING ENABLED] the cpu utilization to scale up to
            new pods on
    :param disable_nginx: (default False) disable nginx inside of the pod (recommended)
    :param gunicorn_workers: (default 1) [MUST BE 1 FOR SPARK ML models TO PREVENT OOM] Number of web workers.
    :param resource_requests_enabled: (default False) whether or not to enable Kubernetes resource requests
    :param resource_limits_enabled: (default False) whether or not to enable Kubernetes resource limits
    :param cpu_request: (default 0.5) [USED IF RESOURCE REQUESTS ENABLED] number of CPU to request
    :param cpu_limit: (default 1) [USED IF RESOURCE LIMITS ENABLED] number of CPU to cap at
    :param memory_request: (default 512Mi) [USED IF RESOURCE REQUESTS ENABLED] amount of RAM to request
    :param memory_limit: (default 2048Mi) [USED IF RESOURCE LIMITS ENABLED] amount of RAM to limit at
    :param expose_external: (default False) whether or not to create Ingress resource to deploy outside of the cluster.

        :NOTE:
            .. code-block:: text

                It is not recommended to create an Ingress resource using this parameter, as your model will be
                deployed with no authorization (and public access). Instead, it is better to deploy your model
                as an internal service, and deploy an authentication proxy (such as https://github.com/oauth2-proxy/oauth2-proxy)
                to proxy traffic to your internal service after authenticating.
    """
    print("Processing...")

    payload = {
        'run_id': run_id or mlflow.active_run().info.run_uuid, 'handler_name': 'DEPLOY_KUBERNETES',
        'service_port': service_port, 'base_replicas': base_replicas, 'autoscaling_enabled': autoscaling_enabled,
        'max_replicas': max_replicas, 'target_cpu_utilization': target_cpu_utilization,
        'disable_nginx': disable_nginx, 'gunicorn_workers': gunicorn_workers,
        'resource_requests_enabled': resource_requests_enabled, 'memory_limit': memory_limit,
        'resource_limits_enabled': resource_limits_enabled, 'cpu_request': cpu_request, 'cpu_limit': cpu_limit,
        'memory_request': memory_request, 'expose_external': expose_external
    }

    return __initiate_job(payload, '/api/rest/initiate')

@_mlflow_patch('undeploy_kubernetes')
def _undeploy_kubernetes(run_id: str):
    """
    Removes a model deployment from Kubernetes. This will delete the Kubernetes deployment and record the event

    :param run_id: specified if overriding the active run
    """
    print("Processing...")

    payload = {
        'run_id': run_id or mlflow.active_run().info.run_uuid, 'handler_name': 'UNDEPLOY_KUBERNETES'
    }

    return __initiate_job(payload, '/api/rest/initiate')


@_mlflow_patch('deploy_database')
def _deploy_db(db_schema_name: str,
               db_table_name: str,
               run_id: str,
               reference_table: Optional[str] = None,
               reference_schema: Optional[str] = None,
               primary_key: Optional[Dict[str, str]] = None,
               df: Optional[Union[SparkDF, PandasDF]] = None,
               create_model_table: Optional[bool] = True,
               model_cols: Optional[List[str]] = None,
               classes: Optional[List[str]] = None,
               library_specific: Optional[Dict[str, str]] = None,
               replace: Optional[bool] = False,
               max_batch_size: Optional[int] = 10000,
               verbose: bool = False) -> None:
    """
    Deploy a trained (currently Spark, Sklearn, Keras or H2O) model to the Database.
    This either creates a new table or alters an existing table in the database (depending on parameters passed)

    :param db_schema_name: (str) the schema name to deploy to.
    :param db_table_name: (str) the table name to deploy to.
    :param run_id: (str) The run_id to deploy the model on. The model associated with this run will be deployed
    :param reference_table: (str) if creating a new table, an alternative to specifying a dataframe is specifying a
        reference table. The column schema of the reference table will be used to create the new table (e.g. MYTABLE)\n
    :param reference_schema: (str) the db schema for the reference table.
    :param primary_key: (Dict) Dictionary of column + SQL datatype to use for the primary/composite key. \n
        * If you are deploying to a table that already exists, it must already have a primary key, and this parameter will be ignored. \n
        * If you are creating the table in this function, you MUST pass in a primary key
    :param df: (Spark or Pandas DF) The dataframe used to train the model \n
                | NOTE: The columns in this df are the ones that will be used to create the table unless specified by model_cols
    :param create_model_table: Whether or not to create the table from the dataframe. Default True. This
                                Will ONLY be used if the table does not exist and a dataframe is passed in
    :param model_cols: (List[str]) The columns from the table to use for the model. If None, all columns in the table
                                        will be passed to the model. If specified, the columns will be passed to the model
                                        IN THAT ORDER. The columns passed here must exist in the table.
    :param classes: (List[str]) The classes (prediction labels) for the model being deployed.\n
                    NOTE: If not supplied, the table will have default column names for each class
    :param library_specific: (dict{str: str}) Prediction options for certain model types: \n
        * Certain model types (specifically Keras and Scikit-learn) support prediction arguments. Here are the options that we support:
            * Scikit-learn
                * predict_call: determines function call for the model. Available: 'predict' (default), 'predict_proba', 'transform'
                * predict_args: passed into the predict call (for Gaussian and Bayesian models). Available: 'return_std', 'return_cov'
            * Keras
                * pred_threshold: prediction threshold for Keras binary classification models. Note: If the model type is Keras, the output layer has 1 node, and pred_threshold is None, you will NOT receive a class prediction, only the output of the final layer (like model.predict()). If you want a class prediction for your binary classification problem, you MUST pass in a threshold.
    If the model does not support these parameters, they will be ignored.

    :param max_batch_size: (int) the max size for the database to batch groups of rows for prediction. Default 10,000.
    :param replace: (bool) whether or not to replace a currently existing model. This param is not yet implemented
    :return: None

    This function creates the following IF you are creating a table from the dataframe\n
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

    A trigger is also created on the deployment table that runs the model after every insert into that table.
    """
    print("Deploying model to database...")

    # database converts all object names to upper case, so we need to as well in our metadata
    db_schema_name=db_schema_name.upper()
    db_table_name=db_table_name.upper()


    # ~ Backwards Compatability ~
    if verbose:
        print("Deprecated Parameter 'verbose'. Use mlflow.watch_job(<job id>) or mlflow.fetch_logs(<job id>) to get"
              " verbose output. Ignoring...", file=stderr)

    if primary_key is not None:
        if isinstance(primary_key, list):
            print("Passing in primary keys as a list of tuples is deprecated. Use dictionary {column name: type}",
                  file=stderr)
            primary_key = dict(primary_key)

    if df is not None:
        if isinstance(df, PandasDF):
            _check_for_splice_ctx() # We will need a splice context to convert to sparkDF
            df_schema = mlflow._splice_context.pandasToSpark(df).schema.json()
        elif isinstance(df, SparkDF):
            df_schema = df.schema.json()
        else:
            raise SpliceMachineException("Dataframe must either be a Pandas or Spark Dataframe")
    else:
        df_schema = None

    payload = {
        'db_table': db_table_name, 'db_schema': db_schema_name, 'run_id': run_id or mlflow.active_run().info.run_uuid,
        'primary_key': primary_key, 'df_schema': df_schema, 'create_model_table': create_model_table,
        'model_cols': model_cols, 'classes': classes, 'library_specific': library_specific, 'replace': replace,
        'handler_name': 'DEPLOY_DATABASE', 'reference_table': reference_table, 'reference_schema': reference_schema, 
        'max_batch_size': max_batch_size
    }

    return __initiate_job(payload, '/api/rest/initiate')

@_mlflow_patch('undeploy_db')
def _undeploy_db(run_id: str, schema_name: str = None, table_name: str = None, drop_table: bool = False):
    """
    Undeploys an mlflow model from a DB table. If schema_name and table_name are not provided, ALL tables that this
    model is deployed to will be undeployed.

    :param run_id: The run_id of the model
    :param schema_name: The schema name of the deployment table to remove. If not set, all tables deployed with this
    model will be removed
    :param table_name: The table name of the deployment table to remove. If not set, all tables deployed with this model
    will be removed
    :param drop_table: Whether to drop the table or not. If False, the table will exist, but the trigger invoking the
    model will be removed only. Default False
    :return: Job ID to track job progress with the mlflow.watch_job function
    """
    payload = {
        'run_id': run_id,
        'db_schema': schema_name.upper() if schema_name else None,
        'db_table': table_name.upper() if table_name else None,
        'drop_table': drop_table,
        'handler_name': 'UNDEPLOY_DATABASE'
    }
    return __initiate_job(payload, '/api/rest/initiate')

def __get_logs(job_id: int):
    """
    Retrieve the logs associated with the specified job id
    """
    request = requests.post(
        get_jobs_uri(mlflow.get_tracking_uri() or get_pod_uri('mlflow', 5003, _testing=_TESTING)) + "/api/rest/logs",
        json={"task_id": job_id}, auth=mlflow._basic_auth
    )
    if not request.ok:
        raise SpliceMachineException(f"Could not retrieve the logs for job {job_id}: {request.status_code}")
    return request.json()['logs']

@_mlflow_patch('watch_job')
def _watch_job(job_id: int):
    """
    Stream the logs in real time to standard out
    of a Job

    :param job_id: the job id to watch (returned after executing an operation)
    :raise SpliceMachineException: If the job being watched fails
    """
    previous_lines = []
    warn = False # If there were any warnings from the log, we want to notify the user explicitly
    while True:
        logs_retrieved = __get_logs(job_id)
        logs_retrieved.remove('')
        log_idx = len(logs_retrieved)
        # searching from the end is faster, because unless the logs double in the interval, it will be closer
        for log_idx in range(len(logs_retrieved) - 1, -1, -1):
            if logs_retrieved[log_idx] in previous_lines:
                break

        idx = log_idx+1 if log_idx else log_idx # First time getting logs, go to 0th index, else log_idx+1
        for n in logs_retrieved[idx:]:
            if 'WARNING' in n:
                warnings.warn(n)
                warn = True
            print(f'\n{n}',end='')

        previous_lines = copy.deepcopy(logs_retrieved)  # O(1) checking
        previous_lines = previous_lines if previous_lines[-1] else previous_lines[:-1] # Remove empty line
        if 'TASK_COMPLETED' in previous_lines[-1]: # Finishing Statement
            # Check for a failure first, and raise an error if so
            for log in reversed(previous_lines):
                if 'ERROR' in log and 'Task Failed' in log:
                    raise SpliceMachineException(
                        'An error occured in your Job. See the log above for more information'
                    ) from None
            if warn:
                print('\n','Note! Your deployment had some warnings you should consider.')
            return

        time.sleep(1)


@_mlflow_patch('fetch_logs')
def _fetch_logs(job_id: int):
    """
    Get the logs as an array
    :param job_id: the job to get the logs for
    """
    return __get_logs(job_id)


@_mlflow_patch('get_deployed_models')
def _get_deployed_models() -> PandasDF:
    """
    Get the currently deployed models in the database
    :return: Pandas df
    """
    print('Rows without a Schema or Table name indicate that the deployment table no longer exists '
          '(based on the TableID). This could be because the table was dropped, or the deployment was DELETED')
    request = requests.get(
        get_jobs_uri(mlflow.get_tracking_uri() or get_pod_uri('mlflow', 5003, _testing=_TESTING)) + "/api/rest/deployments",
        auth=mlflow._basic_auth
    )
    if not request.ok:
        raise SpliceMachineException(request.text)
    return PandasDF(dict(request.json()))


def apply_patches():
    """
    Apply all the Gorilla Patches; \
    All Gorilla Patched MUST be predixed with '_' before their destination in MLflow
    """
    targets = [_register_feature_store, _register_splice_context, _lp, _lm, _timer, _log_artifact, _log_feature_transformations,
               _log_model_params, _log_pipeline_stages, _log_model, _load_model, _download_artifact,
               _start_run, _current_run_id, _current_exp_id, _deploy_aws, _deploy_azure, _deploy_db, _login_director,
               _get_run_ids_by_name, _get_deployed_models, _deploy_kubernetes, _undeploy_kubernetes, _fetch_logs,
               _watch_job, _end_run, _set_mlflow_uri, _remove_active_training_set, _undeploy_db]

    for target in targets:
        gorilla.apply(gorilla.Patch(mlflow, target.__name__.lstrip('_'), target, settings=_GORILLA_SETTINGS))


def _set_mlflow_uri(uri):
    """
    Set the tracking uri for mlflow. Only needed if running outside of the Splice Machine K8s Cloud Service

    :param uri: (str) the URL of your mlflow UI.
    :return: None
    """
    global _CLIENT
    mlflow.set_tracking_uri(uri)
    _CLIENT = mlflow.tracking.MlflowClient(tracking_uri=uri)
    mlflow.client = _CLIENT


def main():
    mlflow.set_tracking_uri(_TRACKING_URL)
    apply_patches()
    __try_auto_login()


main()
