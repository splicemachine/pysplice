from builtins import super
from collections import defaultdict
from os import environ as env_vars, popen as rbash, system as bash
from sys import getsizeof
from time import time, sleep
from enum import Enum
from typing import List, Dict
import re

import requests
from requests.auth import HTTPBasicAuth

import mlflow
import mlflow.spark
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

import pyspark
from pyspark.ml import Pipeline,PipelineModel
from pyspark.ml.base import Model as SparkModel
from py4j.java_gateway import java_import


import mleap.pyspark
from mleap.pyspark.spark_support import SimpleSparkSerializer
from mleap.pyspark import spark_support

from ..spark.context import PySpliceContext

#SimpleSparkSerializer() # Adds the serializeToBundle function from Mleap
CONVERSIONS = PySpliceContext.CONVERSIONS

# For MLeap model deployment
class ModelType(Enum):
    CLASSIFICATION = 0
    REGRESSION = 1
    CLUSTERING_WITH_PROB = 2
    CLUSTERING_WO_PROB = 3

def get_pod_uri(pod, port, pod_count=0, _testing=False):
    """
    Get address of MLFlow Container for Kubernetes
    """

    if _testing:
        url = f"http://{pod}:{port}"  # mlflow docker container endpoint

    elif 'MLFLOW_URL' in env_vars:
        url = env_vars['MLFLOW_URL'].rstrip(':5001')
        url += f':{port}' # 5001 or 5003 for tracking or deployment
    else:
        raise KeyError(
            "Uh Oh! MLFLOW_URL variable was not found... are you running in the Cloud service?")
    return url


def _get_user():
    """
    Get the current logged in user to
    Jupyter
    :return: (str) name of the logged in user
    """
    try:
        uname = env_vars.get('JUPYTERHUB_USER') or env_vars['USER']
        return uname
    except KeyError:
        raise Exception(
            "Could not determine current running user. Running MLManager outside of Splice Machine"
            " Cloud Jupyter is currently unsupported")


def _readable_pipeline_stage(pipeline_stage):
    """
    Get a readable version of the Pipeline stage
    (without the memory address)
    :param pipeline_stage: the name of the stage to parse
    """
    if '_' in str(pipeline_stage):
        return str(pipeline_stage).split('_')[0]
    return str(pipeline_stage)


def _get_stages(pipeline):
    """
    Extract the stages from a fit or unfit pipeline
    :param pipeline: a fit or unfit Spark pipeline
    :return: stages list
    """
    if hasattr(pipeline, 'getStages'):
        return pipeline.getStages()  # fit pipeline
    return pipeline.stages  # unfit pipeline


def _parse_string_parameters(string_parameters):
    """
    Parse string rendering of extractParamMap
    :param string_parameters:
    :return:
    """
    parameters = {}
    parsed_mapping = str(string_parameters).replace("{", "").replace(
        "}", "").replace("\n", "").replace("\t", "").split(',')

    for mapping in parsed_mapping:
        param = mapping[mapping.index('-') + 1:].split(':')[0]
        value = mapping[mapping.index('-') + 1:].split(':')[1]

        parameters[param.strip()] = value.strip()
    return parameters


def _get_cols(transformer, get_input=True):
    """
    Get columns from a transformer
    :param transformer: the transformer (fit or unfit)
    :param get_input: whether or not to return the input columns
    :return: a list of either input or output columns
    """
    col_type = 'Input' if get_input else 'Output'
    if hasattr(transformer, 'get' + col_type + 'Col'):  # single transformer 1:1 (regular)
        col_function = transformer.getInputCol if get_input else transformer.getOutputCol
        return [
            col_function()] if get_input else col_function()  # so we don't need to change code for
        # a single transformer
    elif hasattr(transformer, col_type + "Col"):  # single transformer 1:1 (vec)
        return getattr(transformer, col_type + "Col")
    elif get_input and hasattr(transformer,
                               'get' + col_type + 'Cols'):  # multi ple transformer n:1 (regular)
        return transformer.getInputCols()  # we can never have > 1 output column (goes to vector)
    elif get_input and hasattr(transformer, col_type + 'Cols'):  # multiple transformer n:1 (vec)
        return getattr(transformer, col_type + "Cols")
    else:
        print(
            "Warning: Transformer " + str(
                transformer) + " could not be parsed. If this is a model, this is expected.")

def _get_feature_vector_columns(fittedPipe: PipelineModel) -> List[str]:
    '''
    Gets the input columns from the VectorAssembler stage of a fitted Pipeline
    '''
    vec_stage = None
    for i in fittedPipe.stages:
        if 'VectorAssembler' in str(i):
            vec_stage = i
            break
    return _get_cols(vec_stage)

def _get_model_stage(pipeline):
    '''
    Gets the Model stage of a PipelineModel
    '''
    for i in _get_stages(pipeline):
        # FIXME: We need a better way to determine if a stage is a model
        if 'Model' in str(i.__class__) and i.__module__.split('.')[-1] in ['clustering', 'classification', 'regression']:
            return i
    raise AttributeError('Could not find model stage in Pipeline!')

def _get_num_classes(pipeline_or_model) -> int:
    '''
    Tries to find the number of classes in a Pipeline or Model object
    :param pipeline_or_model: The Pipeline or Model object
    '''
    if 'pipeline' in str(pipeline_or_model.__class__).lower():
        model = _get_model_stage(pipeline_or_model)
    else:
        model = pipeline_or_model
    num_classes = model.numClasses if model.hasParam('numClasses') else model.summary.k
    return num_classes   

def _get_model_type(pipeline_or_model) -> ModelType:
    """
    Takes a fitted Spark Pipeline or Model and determines if it is a Regression, Classification, or Clustering model
    """
    model = _get_model_stage(pipeline_or_model)
        
    if model.__module__   == 'pyspark.ml.classification': m_type = ModelType.CLASSIFICATION
    elif model.__module__ == 'pyspark.ml.regression':     m_type = ModelType.REGRESSION
    elif model.__module__ == 'pyspark.ml.clustering':
        if 'probabilityCol' in model.explainParams(): m_type = ModelType.CLUSTERING_WITH_PROB
        else: m_type = ModelType.CLUSTERING_WO_PROB
    
    return m_type



class MLManager(MlflowClient):
    """
    A class for managing your MLFlow Runs/Experiments
    """ 
    MLMANAGER_SCHEMA = 'MLMANAGER'

    ARTIFACT_INSERT_SQL = f'INSERT INTO {MLMANAGER_SCHEMA}.ARTIFACTS (run_uuid, name, "size", "binary") VALUES (?, ?, ?, ?)'
    ARTIFACT_RETRIEVAL_SQL = 'SELECT "binary" FROM ' + f'{MLMANAGER_SCHEMA}.' + 'ARTIFACTS WHERE name=\'{name}\' ' \
                             'AND run_uuid=\'{runid}\''
    MLEAP_INSERT_SQL = f'INSERT INTO {MLMANAGER_SCHEMA}.MODELS(ID, MODEL) VALUES (?, ?)'
    MLEAP_RETRIEVAL_SQL = 'SELECT MODEL FROM {MLMANAGER_SCHEMA}.MODELS WHERE ID=\'{run_uuid}\''

    def __init__(self, splice_context, tracking_uri=None, _testing=False):
        """
        Tracking URI: the URL for
        :param splice_context: (PySpliceContext) the Python Native Spark Datasource
        :param tracking_uri: MLFlow Tracking Server Endpoint.
            If http based, this must start with http://, or it
            will be assumed as a file store. Defaults to
            MLFlow DC/OS Pod
        """
        self._testing = _testing
        if not tracking_uri:
            server_endpoint = get_pod_uri("mlflow", "5001", _testing=_testing)
        else:
            server_endpoint = tracking_uri

        mlflow.set_tracking_uri(server_endpoint)
        print("Tracking Model Metadata on MLFlow Server @ " + mlflow.get_tracking_uri())

        if not mlflow.get_tracking_uri() == server_endpoint:
            Warning("MLManager doesn't seem to be communicating with the right server endpoint."
                    "Try instantiating this class again!")

        super().__init__(server_endpoint)  # initialize super class

        if _testing:
            self.splice_context = None
        else:
            self.splice_context = splice_context
            java_import(splice_context.jvm, "java.io.{BinaryOutputStream, ObjectOutputStream, ByteArrayInputStream}")

        self.active_run = None
        self.active_experiment = None
        self.timer_start_time = None  # for timer
        self.timer_name = None
        self._basic_auth = None

    @property
    def current_run_id(self):
        """
        Returns the UUID of the current run
    def __repr__(self):
        return "MLManager: Active Experiment " + str(self.active_experiment) + " | Active Run " + str(self.active_run)
    def __str__(self):
        return self.__repr__()
    @staticmethod
    def __removekey(d, key):
        """
        return self.active_run.info.run_uuid

    @property
    def experiment_id(self):
        """
        Returns the UUID of the current experiment
        """
        return self.active_experiment.experiment_id

    def __repr__(self):
        """
        Return String Representation of Current MLManager
        :return:
        """
        return "MLManager\n" + ('-' * 20) + "\nCurrent Experiment: " + \
               str(self.active_experiment) + "\n"

    def __str__(self):
        return self.__repr__()

    def check_active(func):
        """
        Decorator to make sure that run/experiment
        is active
        """

        def wrapped(self, *args, **kwargs):
            if not self.active_experiment:
                raise Exception("Please either use set_active_experiment or create_experiment "
                                "to set an active experiment before running this function")
            elif not self.active_run:
                raise Exception("Please either use set_active_run or start_run to set an active "
                                "run before running this function")
            else:
                return func(self, *args, **kwargs)

        return wrapped

    def lp(self, *args, **kwargs):
        """
        Shortcut function for logging
        parameters
        """
        return self.log_param(*args, **kwargs)

    def lm(self, *args, **kwargs):
        """
        Shortcut function for logging
        metrics
        """
        return self.log_metric(*args, **kwargs)

    def st(self, *args, **kwargs):
        """
        Shortcut function for setting tags
        """
        return self.set_tag(*args, **kwargs)

    def create_experiment(self, experiment_name, reset=False):
        """
        Create a new experiment. If the experiment
        already exists, it will be set to active experiment.
        If the experiment doesn't exist, it will be created
        and set to active. If the reset option is set to true
        (please use with caution), the runs within the existing
        experiment will be deleted
        :param experiment_name: (str) the name of the experiment to create
        :param reset: (bool) whether or not to overwrite the existing run
        """
        experiment = self.get_experiment_by_name(experiment_name)
        if experiment and experiment.lifecycle_stage == 'active':
            print(
                "Experiment " + experiment_name + " already exists... setting to active experiment")
            self.active_experiment = experiment
            print("Active experiment has id " + str(experiment.experiment_id))
            if reset:
                print(
                    "Keyword argument \"reset\" was set to True. "
                    "Overwriting experiment and its associated runs...")
                experiment_id = self.active_experiment.experiment_id
                associated_runs = self.list_run_infos(experiment_id)
                for run in associated_runs:
                    print("Deleting run with UUID " + run.run_uuid)
                    self.delete_run(run.run_uuid)
                print("Successfully overwrote experiment")
        else:
            if experiment and experiment.lifecycle_stage == 'deleted':
                raise Exception(f"Experiment {experiment_name} is deleted. Please choose a new name")
            experiment_id = super(MLManager, self).create_experiment(experiment_name)
            print("Created experiment with id=" + str(experiment_id))
            sleep(2)  # sleep for two seconds to allow rest API to be hit
            self.active_experiment = self.get_experiment_by_name(experiment_name)
            print("Set experiment id=" + str(experiment_id) + " to the active experiment")

    def set_active_experiment(self, experiment_name):
        """
        Set the active experiment of which all new runs will be created under
        Does not apply to already created runs
        :param experiment_name: either an integer (experiment id) or a string (experiment name)
        """

        if isinstance(experiment_name, str):
            self.active_experiment = self.get_experiment_by_name(experiment_name)

        elif isinstance(experiment_name, int):
            self.active_experiment = self.get_experiment(experiment_name)

    def set_active_run(self, run_id):
        """
        Set the active run to a previous run (allows you to log metadata for completed run)
        :param run_id: the run UUID for the previous run
        """
        self.active_run = self.get_run(run_id)

    def start_run(self, tags={}, run_name=None, experiment_id=None, nested=False):
        """
        Create a new run in the active experiment and set it to be active
        :param tags: a dictionary containing metadata about the current run.
            For example:
                {
                    'team': 'pd',
                    'purpose': 'r&d'
                }
        :param run_name: an optional name for the run to show up in the MLFlow UI
        :param experiment_id: if this is specified, the experiment id of this
            will override the active run.
        :param nester: Controls whether run is nested in parent run. True creates a nest run
        """
        if experiment_id:
            new_run_exp_id = experiment_id
            self.set_active_experiment(experiment_id)
        elif self.active_experiment:
            new_run_exp_id = self.active_experiment.experiment_id
        else:
            new_run_exp_id = 0
            try:
                self.set_active_experiment(new_run_exp_id)
            except MlflowException:
                raise MlflowException(
                    "There are no experiements available yet. Please create an experiment before starting a run")

        tags['mlflow.user'] = _get_user()

        self.active_run = super(MLManager, self).create_run(new_run_exp_id, tags=tags)
        if run_name:
            self.set_tag('mlflow.runName', run_name)
            print(f'Setting {run_name} to active run')

    def get_run(self, run_id):
        """
        Retrieve a run (and its data) by its run id
        :param run_id: the run id to retrieve
        """
        try:
            return super(MLManager, self).get_run(run_id)
        except:
            raise Exception(f"The run id {run_id} does not exist. Please check the id")

    @check_active
    def reset_run(self):
        """
        Reset the current run (deletes logged parameters, metrics, artifacts etc.)
        """
        self.delete_run(self.active_run.info.run_uuid)
        self.start_run()

    def __log_param(self, *args, **kwargs):
        super(MLManager, self).log_param(self.active_run.info.run_uuid, *args, **kwargs)

    @check_active
    def log_param(self, *args, **kwargs):
        """
        Log a parameter for the active run
        """
        self.__log_param(*args, **kwargs)

    @check_active
    def log_params(self, params):
        """
        Log a list of parameters in order
        :param params: a list of tuples containing parameters mapped to parameter values
        """
        for parameter in params:
            self.log_param(*parameter)

    def __set_tag(self, *args, **kwargs):
        super(MLManager, self).set_tag(self.active_run.info.run_uuid, *args, **kwargs)

    @check_active
    def set_tag(self, *args, **kwargs):
        """
        Set a tag for the active run
        """
        self.__set_tag(*args, **kwargs)

    @check_active
    def set_tags(self, tags):
        """
        Log a list of tags in order or a dictionary of tags
        :param params: a list of tuples containing tags mapped to tag values
        """
        if isinstance(tags, dict):
            tags = list(tags.items())
        for tag in tags:
            self.set_tag(*tag)

    def __log_metric(self, *args, **kwargs):
        super(MLManager, self).log_metric(self.active_run.info.run_uuid, *args, **kwargs)

    @check_active
    def log_metric(self, *args, **kwargs):
        """
        Log a metric for the active run
        """
        self.__log_metric(*args, **kwargs)

    @check_active
    def log_metrics(self, metrics):
        """
        Log a list of metrics in order
        :param params: a list of tuples containing metrics mapped to metric values
        """
        for metric in metrics:
            self.log_metric(*metric)

    @check_active
    def log_artifact(self, file_name, name):
        """
        Log an artifact for the active run
        :param file_name: (str) the name of the file name to log
        :param name: (str) the name of the run relative name to store the model under
        """
        with open(file_name, 'rb') as artifact:
            byte_stream = bytearray(bytes(artifact.read()))

        self._insert_artifact(name, byte_stream)

    @check_active
    def log_artifacts(self, file_names, names):
        """
        Log artifacts for the active run
        :param file_names: (list) list of file names to retrieve
        :param names: (list) corresponding list of names
            for each artifact in file_names
        """
        for file_name, name in zip(file_names, names):
            self.log_artifact(file_name, name)

    @check_active
    def log_spark_model(self, model, name='model'):
        """
        Log a fitted spark pipeline or model
        :param model: (PipelineModel or Model) is the fitted Spark Model/Pipeline to store
            with the current run
        :param name: (str) the run relative name to store the model under
        """
        if self.active_run.data.tags.get('splice.model_name'):  # this function has already run
            raise Exception("Only one model is permitted per run.")

        self.set_tag('splice.model_name', name)  # read in backend for deployment

        jvm = self.splice_context.jvm

        if self._is_spark_model(model):
            model = PipelineModel(
                stages=[model]
            )  # create a pipeline with only the model if a model is passed in

        baos = jvm.java.io.ByteArrayOutputStream()  # serialize the PipelineModel to a byte array
        oos = jvm.java.io.ObjectOutputStream(baos)
        oos.writeObject(model._to_java())
        oos.flush()
        oos.close()

        self._insert_artifact(name, baos.toByteArray())  # write the byte stream to the db as a BLOB

    @staticmethod
    def _is_spark_model(spark_object):
        """
        Returns whether or not the given
        object is a spark pipeline. If it
        is a model, it will return True, if it is a
        pipeline model is will return False.
        Otherwise, it will throw an exception
        :param spark_object: (Model) Spark object to check
        :return: (bool) whether or not the object is a model
        :exception: (Exception) throws an error if it is not either
        """
        if isinstance(spark_object, PipelineModel):
            return False

        if isinstance(spark_object, SparkModel):
            return True

        raise Exception("The model supplied does not appear to be a Spark Model!")

    def _insert_artifact(self, name, byte_array, mleap_model=False):
        """
        :param name: (str) the path to store the binary
            under (with respect to the current run)
        :param byte_array: (byte[]) Java byte array
        """
        db_connection = self.splice_context.getConnection()
        file_size = getsizeof(byte_array)
        model_ = 'Mleap Model' if mleap_model else 'binary artifact'
        print(f"Saving {model_} of size: {file_size / 1000.0} KB to Splice Machine DB")
        binary_input_stream = self.splice_context.jvm.java.io.ByteArrayInputStream(byte_array)

        if mleap_model:
            prepared_statement = db_connection.prepareStatement(self.MLEAP_INSERT_SQL)
            run_id = name if name else self.current_run_id
            prepared_statement.setString(1, run_id)
            prepared_statement.setBinaryStream(2,binary_input_stream)

        else:
            prepared_statement = db_connection.prepareStatement(self.ARTIFACT_INSERT_SQL)
            prepared_statement.setString(1, self.current_run_id)
            prepared_statement.setString(2, name)
            prepared_statement.setInt(3, file_size)
            prepared_statement.setBinaryStream(4, binary_input_stream)

        prepared_statement.execute()
        prepared_statement.close()

    def __log_batch(self, *args, **kwargs):
        """
        Log batch metrics, parameters, tags
        """
        super(MLManager, self).log_batch(self.active_run.info.run_uuid, *args, **kwargs)

    @check_active
    def log_batch(self, metrics, parameters, tags):
        """
        Log a batch set of metrics, parameters and tags
        :param metrics: a list of tuples mapping metrics to metric values
        :param parameters: a list of tuples mapping parameters to parameter values
        :param tags: a list of tuples mapping tags to tag values
        """
        self.__log_batch(metrics, parameters, tags)

    @check_active
    def log_pipeline_stages(self, pipeline):
        """
        Log the human-friendly names of each stage in
        a  Spark pipeline.
        *Warning*: With a big pipeline, this could result in
        a lot of parameters in MLFlow. It is probably best
        to log them yourself, so you can ensure useful tracking
        :param pipeline: the fitted/unfit pipeline object
        """

        for stage_number, pipeline_stage in enumerate(_get_stages(pipeline)):
            readable_stage_name = _readable_pipeline_stage(pipeline_stage)
            self.log_param('Stage' + str(stage_number), readable_stage_name)

    @staticmethod
    def _find_inputs_by_output(dictionary, value):
        """
        Find the input columns for a given output column
        :param dictionary: dictionary to search
        :param value: column
        :return: None if not found, otherwise first column
        """
        keys = []
        for key in dictionary:
            if dictionary[key][1] == value:  # output column is always the last one
                keys.append(key)
        return keys if len(keys)>0 else None

    @check_active
    def log_feature_transformations(self, unfit_pipeline):
        """
        Log the preprocessing transformation sequence
        for every feature in the UNFITTED Spark pipeline
        :param unfit_pipeline: UNFITTED spark pipeline!!
        """
        transformations = defaultdict(lambda: [[], None])  # transformations, outputColumn

        for stage in _get_stages(unfit_pipeline):
            input_cols, output_col = _get_cols(stage, get_input=True), _get_cols(stage,
                                                                                 get_input=False)
            if input_cols and output_col:  # make sure it could parse transformer
                for column in input_cols:
                    first_column_found = self._find_first_input_by_output(transformations, column)
                    if first_column_found:  # column is not original
                        for f in first_column_found:
                            transformations[f][1] = output_col
                            transformations[f][0].append(
                                _readable_pipeline_stage(stage))
                    else:
                        transformations[column][1] = output_col
                        transformations[column][0].append(_readable_pipeline_stage(stage))

        for column in transformations:
            param_value = ' -> '.join([column] + transformations[column][0] +
                                      [transformations[column][1]])
            self.log_param('Column- ' + column, param_value)

    @check_active
    def start_timer(self, timer_name):
        """
        Start a given timer with the specified
        timer name, which will be logged when the
        run is stopped
        :param timer_name: the name to call the timer (will appear in MLFlow UI)
        """
        self.timer_name = timer_name
        self.timer_start_time = time()

        print("Started timer " + timer_name + " at " + str(self.timer_start_time))

    @check_active
    def log_and_stop_timer(self):
        """
        Stop any active timers, and log the
        time that the timer was active as a parameter
        :return: total time in ms
        """
        if not self.timer_name or not self.timer_start_time:
            raise Exception("You must create a timer with start_timer(timer_name) before stopping")
        total_time = (time() - self.timer_start_time) * 1000
        print("Timer " + self.timer_name + " ran for " + str(total_time) + " ms")
        self.log_param(self.timer_name, str(total_time) + " ms")

        self.timer_name = None
        self.timer_start_time = None
        return total_time

    @check_active
    def log_evaluator_metrics(self, splice_evaluator):
        """
        Takes an Splice evaluator and logs
        all of the associated metrics with it
        :param splice_evaluator: a Splice evaluator (from
            splicemachine.ml.utilities package in pysplice)
        :return: retrieved metrics dict
        """
        results = splice_evaluator.get_results('dict')
        for metric in results:
            self.log_metric(metric, results[metric])

    @check_active
    def log_model_params(self, pipeline_or_model):
        """
        Log the parameters of a fitted model or a
        model part of a fitted pipeline
        :param pipeline_or_model: fitted pipeline/fitted model
        """

        model = _get_model_stage(pipeline_or_model)

        self.log_param('model', _readable_pipeline_stage(model))
        if hasattr(model, '_java_obj'):
            verbose_parameters = _parse_string_parameters(model._java_obj.extractParamMap())
        elif hasattr(model, 'getClassifier'):
            verbose_parameters = _parse_string_parameters(
                model.getClassifier()._java_obj.extractParamMap())
        else:
            raise Exception("Could not parse model type: " + str(model))
        for param in verbose_parameters:
            try:
                value = float(verbose_parameters[param])
                self.log_param('Hyperparameter- ' + param.split('-')[0], value)
            except:
                self.log_param('Hyperparameter- ' + param.split('-')[0], verbose_parameters[param])

    @check_active
    def end_run(self, create=False, metadata={}):
        """
        Terminate the current run
        """
        self.active_run = None

        if create:
            self.start_run(metadata)

    @check_active
    def delete_active_run(self):
        """
        Delete the current run
        """
        self.delete_run(self.active_run.info.run_uuid)
        self.active_run = None

    def retrieve_artifact_stream(self, run_id, name):
        """
        Retrieve the binary stream for a given
        artifact with the specified name and run id
        :param run_id: (str) the run id for the run
            that the artifact belongs to
        :param name: (str) the name of the artifact
        :return: (bytearray(byte)) byte array from BLOB
        """
        try:
            return self.splice_context.df(
                self.ARTIFACT_RETRIEVAL_SQL.format(name=name, runid=run_id)
            ).collect()[0][0]
        except IndexError as e:
            raise Exception(f"Unable to find the artifact with the given run id {run_id} and name {name}")

    def download_artifact(self, name, local_path, run_id=None):
        """
        Download the artifact at the given
        run id (active default) + name
        to the local path
        :param name: (str) artifact name to load
            (with respect to the run)
        :param local_path: (str) local path to download the
            model to
        :param run_id: (str) the run id to download the artifact
            from. Defaults to active run
        """
        blob_data = self.retrieve_artifact_stream(run_id, name)
        with open(local_path, 'wb') as artifact_file:
            artifact_file.write(blob_data)

    def load_spark_model(self, run_id=None, name='model'):
        """
        Download a model from database
        and load it into Spark
        :param run_id: the id of the run to get a model from
            (the run must have an associated model with it named spark_model)
        """
        if not run_id:
            run_id = self.current_run_id
        else:
            run_id = self.get_run(run_id).info.run_uuid

        spark_pipeline_blob = self.retrieve_artifact_stream(run_id, name)
        bis = self.splice_context.jvm.java.io.ByteArrayInputStream(spark_pipeline_blob)
        ois = self.splice_context.jvm.java.io.ObjectInputStream(bis)
        pipeline = PipelineModel._from_java(ois.readObject())  # convert object from Java
        # PipelineModel to Python PipelineModel
        ois.close()

        if len(pipeline.stages) == 1 and self._is_spark_model(pipeline.stages[0]):
            pipeline = pipeline.stages[0]

        return pipeline

    def login_director(self, username, password):
        """
        Login to MLmanager Director so we can
        submit jobs
        :param username: (str) database username
        :param password: (str) database password
        """
        self._basic_auth = HTTPBasicAuth(username, password)


    #######################JOBS#########################
    def _initiate_job(self, payload, endpoint):
        """
        Send a job to the initiation endpoint
        :param payload: (dict) JSON payload for POST request
        :param endpoint: (str) REST endpoint to target
        :return: (str) Response text from request
        """
        if not self._basic_auth:
            raise Exception(
                "You have not logged into MLManager director."
                " Please run manager.login_director(username, password)"
            )
        request = requests.post(
            get_pod_uri('mlflow', 5003, _testing=self._testing) + endpoint,
            auth=self._basic_auth,
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

    def deploy_aws(self, app_name,
                   region='us-east-2', instance_type='ml.m5.xlarge',
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
        sleep(3)  # give the mlflow server time to register the artifact, if necessary

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
            'handler_name': 'DEPLOY_AWS', 'run_id': run_id if run_id else self.current_run_id,
            'region': region, 'user': _get_user(),
            'instance_type': instance_type, 'instance_count': instance_count,
            'deployment_mode': deployment_mode, 'app_name': app_name
        }

        return self._initiate_job(request_payload, '/api/rest/initiate')

    def toggle_service(self, service_name, action):
        """
        Run a modifier on a service
        :param service_name: (str) the service to modify
        :param action: (str) the action to execute
        :return: (str) response text from POST request
        """
        supported_services = ['DEPLOY_AWS', 'DEPLOY_AZURE']
        supported_actions = ['ENABLE_SERVICE', 'DISABLE_SERVICE']
        action = action.upper()  # capitalize, as db likes ca
        if service_name not in supported_services:
            raise Exception('Service must be in list: ' + str(supported_services))
        if action not in supported_actions:
            raise Exception('Service must be in list: ' + str(supported_actions))

        request_payload = {
            'handler_name': action,
            'service': service_name
        }

        return self._initiate_job(request_payload, '/api/rest/initiate')

    def enable_service(self, service_name):
        """
        Enable a given service
        :param service_name: (str) service to enable
        """
        self.toggle_service(service_name, 'ENABLE_SERVICE')

    def disable_service(self, service_name):
        """
        Disable a given service
        :param service_name: (str) service to disable
        """
        self.toggle_service(service_name, 'DISABLE_SERVICE')

    def deploy_azure(self, endpoint_name, resource_group, workspace, run_id=None, region='East US',
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
            'run_id': run_id if run_id else self.current_run_id,
            'cpu_cores': cpu_cores,
            'allocated_ram': allocated_ram,
            'model_name': model_name
        }
        return self._initiate_job(request_payload, '/api/rest/initiate')

        ################MLEAP#################
    def _get_mleap_model(self, fittedPipe, df, run_id: str):
        """
        Turns a fitted Spark Pipeline into an Mleap Transformer
        :param fittedPipe: Fitted Spark Pipeline
        :param df: A TRANSFORMED dataframe. ie a dataframe that the pipeline has called .transform() on
        :param run_id: (str) the MLFlow run associated with the model
        """
        SimpleSparkSerializer() # Adds the serializeToBundle function from Mleap
        if 'tmp' not in rbash('ls /').read():
            bash('mkdir /tmp')
        #Serialize the Spark model into Mleap format
        if f'{run_id}.zip' in rbash('ls /tmp').read():
            bash(f'rm /tmp/{run_id}.zip')
        fittedPipe.serializeToBundle(f"jar:file:///tmp/{run_id}.zip", df)
        
        jvm = self.splice_context.jvm
        java_import(jvm, "com.splicemachine.mlrunner.FileRetriever")
        obj = jvm.FileRetriever.loadBundle(f'jar:file:///tmp/{run_id}.zip')
        bash(f'rm /tmp/{run_id}.zip"')
        return obj

    def insert_mleap_model(self, run_id: str, model) -> None:
        """
        Insert an MLeap Transformer model into the database as a Blob
        :param model_id: (str) the mlflow run_id that the model is associated with
            (with respect to the current run)
        :param model: (Transformer) the fitted Mleap Transformer (pipeline)
        """
        
        #If a model with this run_id already exists in the table, gracefully fail
        #May be faster to use prepared statement
        model_exists = self.splice_context.df(f'select count(*) from {self.MLMANAGER_SCHEMA}.models where ID=\'{run_id}\'').collect()[0][0]
        if model_exists:
            print('A model with this ID already exists in the table. We are NOT replacing it. We will use the currently existing model.\nTo replace, use a new run_id')

        else:
            #Serialize Mleap model to BLOB
            baos = self.splice_context.jvm.java.io.ByteArrayOutputStream() 
            oos = self.splice_context.jvm.java.io.ObjectOutputStream(baos)
            oos.writeObject(model)
            oos.flush()
            oos.close()
            byte_array = baos.toByteArray()
            self._insert_artifact(run_id, byte_array, mleap_model=True)
    
    def __create_data_table(self, schema_table_name: str,schema_str: str) -> str:
        """
        Creates the table that holds the columns of the feature vector as well as a unique MOMENT_ID
        :param schema_table_name: (str) the schema.table to create the table under
        :param schema_str: (str) the structure of the schema of the table as a string (col_name TYPE,)
        """
        if self.splice_context.tableExists(schema_table_name):
            raise NotImplementedError('A model has already been deployed to this table. We currently only support deploying 1 model per table')
        SQL_TABLE = f'CREATE TABLE {schema_table_name} (\n'
        SQL_TABLE += schema_str + '\tMOMENT_ID VARCHAR(150) PRIMARY KEY)'
        self.splice_context.execute(SQL_TABLE)
        return SQL_TABLE
    
    def __create_data_preds_table(self, schema_table_name: str, classes: List[str], modelType: ModelType) -> str:
        """
        Creates the data prediction table that holds the prediction for the rows of the data table
        :param schema_table_name: (str) the schema.table to create the table under
        :param classes: (List[str]) the labels of the model (if they exist)
        :param modelType: (ModelType) Whether the model is a Regression, Classification or Clustering (with/without probabilities)

        Regression models output a DOUBLE as the prediction field with no probabilities
        Classification models and Certain Clustering models have probabilities associated with them, so we need to handle the extra columns holding those probabilities
        Clustering models without probabilities return only an INT.
        The database is strongly typed so we need to specify the output type for each ModelType
        """
        SQL_PRED_TABLE = f'''CREATE TABLE {schema_table_name}_PREDS (
        \tCUR_USER VARCHAR(50) DEFAULT CURRENT_USER,
        \tEVAL_TIME TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        \tMOMENT_ID VARCHAR(250) PRIMARY KEY,
        
        '''   
        if modelType == ModelType.REGRESSION:
            SQL_PRED_TABLE += '\tPREDICTION DOUBLE)'
        
        elif modelType in (ModelType.CLASSIFICATION, ModelType.CLUSTERING_WITH_PROB):
            SQL_PRED_TABLE += '\tPREDICTION VARCHAR(250),'
            for i in classes:
                SQL_PRED_TABLE += f'\t{i} DOUBLE,\n'
            SQL_PRED_TABLE = SQL_PRED_TABLE.rstrip('\n,') + ')'
        
        elif modelType == ModelType.CLUSTERING_WO_PROB:
            SQL_PRED_TABLE += '\tPREDICTION INT)'
        
        self.splice_context.execute(SQL_PRED_TABLE)
        return SQL_PRED_TABLE
    
    def __create_prediction_trigger(self, schema_table_name: str, run_id: str, feature_columns: List[str], schema_types: Dict[str, str], schema_str: str, modelType: ModelType) -> str:
        """
        Creates the trigger that calls the model on data row insert. This trigger will call predict when a new row is inserted into the data table 
        and insert the result into the predictions table.
        :param schema_table_name: (str) the schema.table to create the table under
        :param run_id: (str) the run_id to deploy the model under
        :param feature_columns: (List[str]) the features in the feature vector
        :param schema_types: (Dict[str, str]) a mapping of feature column to data type
        :param schema_str: (str) the structure of the schema of the table as a string (col_name TYPE,)
        :param classes: (List[str]) the labels of the model (if they exist)
        :param modelType: (ModelType) Whether the model is a Regression, Classification or Clustering (with/without probabilities)
        """
    
        if modelType == ModelType.CLASSIFICATION: prediction_call = 'PREDICT_CLASSIFICATION'
        elif modelType == ModelType.REGRESSION: prediction_call = 'PREDICT_REGRESSION'
        elif modelType == ModelType.CLUSTERING_WITH_PROB: prediction_call = 'PREDICT_CLUSTER_PROBABILITIES'
        else: prediction_call = 'PREDICT_CLUSTER'

        SQL_PRED_TRIGGER = f'CREATE TRIGGER runModel_{run_id}\n \tAFTER INSERT\n \tON {schema_table_name}\n \tREFERENCING NEW AS NEWROW\n \tFOR EACH ROW\n \t\tINSERT INTO {schema_table_name}_PREDS(MOMENT_ID,PREDICTION)' \
                        f'\n\t\tVALUES(NEWROW.MOMENT_ID,{prediction_call}(\'{run_id}\','
        
        for i,col in enumerate(feature_columns):
            SQL_PRED_TRIGGER += '||' if i != 0 else ''
            inner_cast = f'CAST(NEWROW.{col} as DECIMAL(38,10))' if schema_types[str(col)] in {'FloatType','DoubleType', 'DecimalType'} else f'NEWROW.{col}'
            SQL_PRED_TRIGGER += f'TRIM(CAST({inner_cast} as CHAR(41)))||\',\''
        
        #Cleanup + schema for PREDICT call
        SQL_PRED_TRIGGER = SQL_PRED_TRIGGER[:-5].lstrip('||') + ',\n\'' + schema_str.replace('\t','').replace('\n','').rstrip(',') + '\'))'
        self.splice_context.execute(SQL_PRED_TRIGGER.replace('\n',' ').replace('\t',' '))
        return SQL_PRED_TRIGGER
    
    def __create_parsing_trigger(self, schema_table_name: str, run_id: str, classes: List[str]) -> str:
        """
        Creates the secondary trigger that parses the results of the first trigger and updates the prediction row populating the relevant columns
        :param schema_table_name: (str) the schema.table to create the table under
        :param run_id: (str) the run_id to deploy the model under
        :param classes: (List[str]) the labels of the model (if they exist)
        """
        SQL_PARSE_TRIGGER = f'CREATE TRIGGER PARSERESULT_{run_id}\n \tAFTER INSERT\n \tON {schema_table_name}_PREDS\n \tREFERENCING NEW AS NEWROW\n \tFOR EACH ROW\n \t\tUPDATE {schema_table_name}_PREDS set '
        case_str = 'PREDICTION=\n\t\tCASE\n'
        for i,c in enumerate(classes):
            SQL_PARSE_TRIGGER += f'{c}=PARSEPROBS(NEWROW.prediction,{i}),'
            case_str += f'\t\tWHEN GETPREDICTION(NEWROW.prediction)={i} then \'{c}\'\n'
        case_str += '\t\tEND WHERE MOMENT_ID=NEWROW.MOMENT_ID'
        SQL_PARSE_TRIGGER += case_str
        self.splice_context.execute(SQL_PARSE_TRIGGER.replace('\n',' ').replace('\t',' '))
        return SQL_PARSE_TRIGGER

    def deploy_model(self, fittedPipe, df, run_id: str = None, db_schema_name: str=None, db_table_name: str=None, classes:List[str]=[], verbose: bool=False, replace=False) -> None:
        """
        Function to deploy a trained (Spark for now) model to the Database. This creates 2 tables: One with the features of the model, and one with the prediction and metadata. 
        They are linked with a column called MOMENT_ID
        
        :param fittedPipe: (spark pipeline or model) The fitted pipeline to deploy
        :param df: (Spark DF) The dataframe used to train the model
        :param run_id: (str) The active run_id
        :param db_schema_name: (str) the schema name to deploy to. If None, the currently set schema will be used.
        :param db_table_name: (str) the table name to deploy to. If none, the run_id will be used for the table name(s)
        :param classes: List[str] The classes (prediction values) for the model being deployed. NOTE: If not supplied, the table will have column named c0,c1,c2 etc for each class
        :param
        
        This function creates the following:
        * Table called DATA_{run_id} where run_id is the run_id of the mlflow run associated to that model. This will have a column for each feature in the feature vector as well as a MOMENT_ID as primary key
        * Table callsed DATA_PRED_{run_id} That will have the columns:
            USER which is the current user who made the request
            EVAL_TIME which is the CURRENT_TIMESTAMP
            MOMENT_ID same as the DATA table to link predictions to rows in the table
            PREDICTION. The prediction of the model. If the :classes: param is not filled in, this will be c0,c1,c2 etc for classification models
            A column for each class of the predictor with the value being the probability/confidence of the model
        * A trigger that runs on (after) insertion to the data table that runs an INSERT into the prediction table, 
            calling the PREDICT function, passing in the row of data as well as the schema of the dataset, and the run_id of the model to run
        * A trigger that runs on (after) insertion to the prediction table that calls an UPDATE to the row inserted, parsing the prediction probabilities and filling in proper column values
            
        """
        
        #See if the df passed in has already been transformed.
        #If not, transform it
        if 'prediction' not in df.columns:
            df = fittedPipe.transform(df)
        
        run_id = run_id if run_id else self.current_run_id
        db_table_name = db_table_name if db_table_name else f'data_{run_id}'
        schema_table_name = f'{db_schema_name}.{db_table_name}' if db_schema_name else db_table_name
        
        #THIS WILL BE CREATED AUTOMATICALLY ON MLFLOW POD CREATION
        if not self.splice_context.tableExists('models'):
            self.splice_context.execute(f'create table {self.MLMANAGER_SCHEMA}.models (id varchar(100) PRIMARY KEY, model BLOB)')
        
        #Get model type
        modelType = _get_model_type(fittedPipe)
        
        print(f'Deploying model {run_id} to table')
        
        if classes:
            if modelType not in (ModelType.CLASSIFICATION, ModelType.CLUSTERING_WITH_PROB):
                print('Prediction labels found but model is not type Classification. Removing labels')
                classes = None
            else:
                #handling spaces in class names
                classes = [f'\"{c}\"' if ' ' in c else c for c in classes]
                print(f'Prediction labels found. Using {classes} as labels for predictions {list(range(0, len(classes)))} respectively')
        else:
            if modelType in (ModelType.CLASSIFICATION, ModelType.CLUSTERING_WITH_PROB):
                #Add a column for each class of the prediction to output the probability of the prediction
                classes = [f'C{i}' for i in range(_get_num_classes(fittedPipe))]
            
        #Get the Mleap model and insert it into the MODELS table
        mleap_model = self._get_mleap_model(fittedPipe, df, run_id)
        self.insert_mleap_model(run_id, mleap_model)
        
        #Get the VectorAssembler so we can get the features of the model
        feature_columns = _get_feature_vector_columns(fittedPipe)
        #Get the datatype of each column in the dataframe
        schema_types = {str(i.name): re.sub("[0-9,()]", "",str(i.dataType)) for i in df.schema}
        
        #Create the schema of the table (we use this a few times)
        schema_str = ''
        for i in feature_columns:
            schema_str += f'\t{i} {CONVERSIONS[schema_types[str(i)]]},\n'
        
        
        #Create table 1: DATA
        print('Creating data table ...', end=' ')
        SQL_TABLE_STR = self.__create_data_table(schema_table_name, schema_str)
        print('Done.')
        
        #Create table 2: DATA_PREDS
        print('Creating prediction table ...', end=' ')
        SQL_PRED_TABLE_STR = self.__create_data_preds_table(schema_table_name, classes, modelType)
        print('Done.')
        
        
        #Create Trigger 1: (model prediction)
        print('Creating model prediction trigger ...', end=' ')
        SQL_PRED_TRIGGER_STR = self.__create_prediction_trigger(schema_table_name, run_id, feature_columns, schema_types,schema_str, modelType)
        print('Done.')
        
        if modelType in (ModelType.CLASSIFICATION, ModelType.CLUSTERING_WITH_PROB):
            #Create Trigger 2: model parsing
            print('Creating parsing trigger ...', end=' ')
            SQL_PARSE_TRIGGER_STR = self.__create_parsing_trigger(schema_table_name,run_id, classes)
            print('Done.')

        print('Model Deployed')
        
        if verbose:
            print(SQL_TABLE_STR,end='\n\n')
            print(SQL_PRED_TABLE_STR,end='\n\n')
            print(SQL_PRED_TRIGGER_STR.replace('\n',' ').replace('\t',' '), end='\n\n')
            if modelType in (ModelType.CLASSIFICATION, ModelType.CLUSTERING_WITH_PROB):
                print(SQL_PARSE_TRIGGER_STR)
