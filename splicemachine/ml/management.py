from collections import defaultdict
from time import time, sleep
import os
from hashlib import md5
import requests
import random

import mlflow
import mlflow.h2o
import mlflow.sklearn
import mlflow.spark
from mlflow.tracking import MlflowClient


TESTING = True

def get_pod_uri(pod, port, pod_count=0):
    """
    Get address of MLFlow Container (this is
    for DC/OS (Mesosphere) setup, not Kubernetes
    """
    
    if TESTING:
        return "http://{pod}:{port}".format(pod=pod, port=port) # mlflow docker container endpoint
    
    try:
        return 'http://{pod}-{pod_count}-node.{framework}.mesos:{port}'.format \
            (pod=pod, pod_count=pod_count, framework=os.environ['FRAMEWORK_NAME'], port=port)
        # mlflow pod in mesos endpoint (in production)
    except KeyError as e:
        raise KeyError(
            "Uh Oh! FRAMEWORK_NAME variable was not found... are you running in Zeppelin?")


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
    if hasattr(pipeline, 'stages') and isinstance(pipeline.stages, list):
        return pipeline.stages  # fit pipeline
    return pipeline.getStages()  # unfit pipeline


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
            col_function()] if get_input else col_function()  # so we don't need to change code for a single transformer
    elif hasattr(transformer, col_type + "Col"):  # single transformer 1:1 (vec)
        return getattr(transformer, col_type + "Col")
    elif get_input and hasattr(transformer,
                               'get' + col_type + 'Cols'):  # multi ple transformer n:1 (regular)
        return transformer.getInputCols()  # we can never have > 1 output column (goes to vector)
    elif get_input and hasattr(transformer, col_type + 'Cols'):  # multiple transformer n:1 (vec)
        return getattr(transformer, col_type + "Cols")
    else:
        print("Warning: Transformer " + str(transformer) + " could not be parsed. If this is a model, this is expected.")


class MLManager(MlflowClient):
    """
    A class for managing your MLFlow Runs/Experiments
    """

    def __init__(self, _tracking_uri=get_pod_uri("mlflow", "5001")):
        mlflow.set_tracking_uri(_tracking_uri)
        print("Tracking Model Metadata on MLFlow Server @ " + mlflow.get_tracking_uri())

        if not mlflow.get_tracking_uri() == _tracking_uri:
            Warning("MLManager doesn't seem to be communicating with the right server endpoint."
                    "Try instantiating this class again!")

        MlflowClient.__init__(self, _tracking_uri)

        self.active_run = None
        self.active_experiment = None

        self.timer_start_time = None  # for timer
        self.timer_name = None
        
        self._vars = {} # for parametrized runs
        

    @property
    def current_run_id(self):
        """
        Returns the UUID of the current run
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
                raise Exception("Please either use set_active_run or create_run to set an active "
                                "run before running this function")
            else:
                return func(self, *args, **kwargs)
        return wrapped
    
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
                    "Keyword argument \"reset\" was set to True. Overwriting experiment and its associated runs...")
                experiment_id = self.active_experiment.experiment_id
                associated_runs = self.list_run_infos(experiment_id)
                for run in associated_runs:
                    print("Deleting run with UUID " + run.run_uuid)
                    self.delete_run(run.run_uuid)
                print("Successfully overwrote experiment")
        else:
            if experiment and experiment.lifecycle_stage == 'deleted':
                raise Exception("Experiment {} is deleted. Please choose a new name".format(experiment_name))
            experiment_id = super(MLManager, self).create_experiment(experiment_name)
            print("Created experiment w/ id=" + str(experiment_id))
            sleep(2) # sleep for two seconds to allow rest API to be hit
            self.active_experiment = self.get_experiment_by_name(experiment_name)

    def set_active_experiment(self, experiment_name):
        """
        Set the active experiment of which all new runs will be created under
        Does not apply to already created runs

        :param experiment_name: either an integer (experiment id) or a string (experiment name)
        """

        if isinstance(experiment_name, str):
            self.active_experiment = self.get_experiment_by_name(experiment_name)

        elif isinstance(experiment_name, int) or isinstance(experiment_name, long):
            self.active_experiment = self.get_experiment(experiment_name)

    def create_new_run(self, run_metadata={}):
        """
        Create a new run in the active experiment and set it to be active
        :param metadata: a dictionary containing metadata about the current run.
            For example:
            {
                "user_id": "john",
                "team": "product development"
            }
        """

        self.create_run(run_metadata)

    def create_run(self, run_metadata={}):
        """
        Create a new run in the active experiment and set it to be active
        :param metadata: a dictionary containing metadata about the current run.
            For example:
            {
                "user_id": "john",
                "team": "product development"
            }
        """
        if not self.active_experiment:
            raise Exception(
                "You must set an experiment before you can create a run."
                " Use MLFlowManager.set_active_experiment")

        self.active_run = super(MLManager, self).create_run(self.active_experiment.experiment_id)

        for key in run_metadata:
            self.set_tag(key, run_metadata[key])
    
    def get_run(run_id):
        """
        Retrieve a run (and its data) by its run id
        :param run_id: the run id to retrieve
        """
        try:
            return super(MLManager, self).get_run(run_id)
        except:
            raise Exception("The run id {run_id} does not exist. Please check the id".format(run_id=run_id))
            
    @check_active
    def reset_run(self):
        """
        Reset the current run (deletes logged parameters, metrics, artifacts etc.)
        """
        self.delete_run(self.active_run.info.run_uuid)
        self.create_new_run()

    @check_active
    def set_active_run(self, run_id):
        """
        Set the active run to a previous run (allows you to log metadata for completed run)
        :param run_id: the run UUID for the previous run
        """
        self.active_run = self.get_run(run_id)

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
        Log a list of tags in order
        :param params: a list of tuples containing tags mapped to tag values
        """
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

    def __log_artifact(self, *args, **kwargs):
        super(MLManager, self).log_artifact(self.active_run.info.run_uuid, *args, **kwargs)
    
    
    def log_artifact(self, *args, **kwargs):
        """
        Log an artifact for the active run
        """
        self.__log_artifact(*args, **kwargs)

    def __log_artifacts(self, *args, **kwargs):
        super(MLManager, self).log_artifacts(self.active_run.info.run_uuid, *args, **kwargs)
    
    @check_active
    def log_artifacts(self, *args, **kwargs):
        """
        Log artifacts for the active run
        """
        self.__log_artifacts(*args, **kwargs)
    
    @check_active
    def log_model(self, model, module):
        """
        Log a model for the active run
        :param model: the fitted model/pipeline (in spark) to log
        :param module: the module that this is part of (mlflow.spark, mlflow.sklearn etc)
        """
        try:
            mlflow.end_run()
        except:
            pass

        with mlflow.start_run(run_id=self.active_run.info.run_uuid):
            module.log_model(model, "spark_model")
    
    @check_active
    def log_spark_model(self, model):
        """
        Log a spark pipeline
        :param model: the fitted pipeline to log
        """
        raise Exception("Error! This has been renamed to log_spark_pipeline. "
              "This function, log_spark_model will be removed in the near future")
        
    @check_active
    def log_spark_pipeline(self, pipeline):
        """
        Log a spark pipeline 
        :param pipeline: the fitted pipeline to log
        """
        self.log_model(pipeline, mlflow.spark)
        
    @check_active
    def log_spark_pipelines(self, pipelines):
        """
        Log a list of spark models in order
        :param models: a list of spark models (fitted) to log
        """
        for pipeline in pipelines:
            self.log_spark_pipeline(model)

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
    def _find_first_input_by_output(dictionary, value):
        """
        Find the first input column for a given column

        :param dictionary: dictionary to search
        :param value: column
        :return: None if not found, otherwise first column
        """
        for key in dictionary:
            if dictionary[key][1] == value:  # output column is always the last one
                return key
        return None

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
                        transformations[first_column_found][1] = output_col
                        transformations[first_column_found][0].append(
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
    def log_model_params(self, pipeline_or_model, stage_index=-1):
        """
        Log the parameters of a fitted model or a
        model part of a fitted pipeline
        :param pipeline_or_model: fitted pipeline/fitted model
        :param stage_index
        """

        if 'pipeline' in str(pipeline_or_model).lower():
            model = pipeline_or_model.stages[stage_index]
        else:
            model = pipeline_or_model

        self.log_param('model', _readable_pipeline_stage(model))
        verbose_parameters = _parse_string_parameters(model._java_obj.extractParamMap())
        parameters = {}
        for param in verbose_parameters:
            try:
                value = float(verbose_parameters[param])
                self.log_metric('Hyperparameter- ' + param.split('-')[0], value)
            except:
                self.log_param('Hyperparameter- ' + param.split('-')[0], verbose_parameters[param])
    
    @check_active
    def terminate(self, create=False, metadata={}):
        """
        Terminate the current run
        """
        self.active_run = None

        if create:
            self.create_new_run(metadata)
    
    @check_active
    def delete_active_run(self):
        """
        Delete the current run
        """
        self.delete_run(self.active_run.info.run_uuid)
        self.active_run = None
        
    def load_model(run_id, module=mlflow.spark):
        """
        Download a model from S3 
        and load it into Spark
        :param run_id: the id of the run to get a model from
            (the run must have an associated model with it named spark_model)
        """
        print("Retrieving Model...")
        run = self.get_run(run_id)
        artifact_location = run.info.artifact_uri
        return module.load_model(artifact_location + "/spark")
    
    @check_active
    def set_vars(self, vars_dictionary):
        """
        Set variables for each variable
        in dictionary. This is for parametrized
        runs. If you use this option, all of
        the current variables will be replaced
        with the ones specified in the dictionary
        
        :param vars_dictionary: dictionary of variables
            mapped to their values. e.g.
            {
                "a": "b",
                "c": "d"
            }
        
        """
        self._vars = vars_dictionary
    
    def get_vars(self):
        """
        Get all of the current variables
        
        """
    
    def deploy_run_sagemaker(self, run_id, app_name, 
                             region='us-east-2', instance_type='ml.m4.xlarge',
                             instance_count=1, deployment_mode='create'):
        """
        Queue Job to deploy a run to sagemaker with the
        given run id (found in MLFlow UI or through search API)
        
        :param run_id: the id of the run to deploy
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
        sleep(3) # give the mlflow server time to register the artifact, if necessary
        
        run = self.get_run(run_id)
        experiment_id = run._info.experiment_id
            
        supported_aws_regions = ['us-east-2', 'us-west-1', 'us-west-2', 'eu-central-1']
        supported_instance_types = ['ml.m4.xlarge']
        supported_deployment_modes = ['create', 'replace', 'add']
        # data validation
        if not region in supported_aws_regions:
            raise Exception("Region must be in list: " + str(supported_aws_regions))
        if not instance_type in supported_instance_types:
            raise Exception("Instance type must be in list: " + str(instance_type))
        if not deployment_mode in supported_deployment_modes:
            raise Exception("Deployment mode must be in list: " + str(supported_deployment_modes))
            
        request_payload = {
            'handler': 'deploy', 'experiment_id': experiment_id, 'run_id': run_id,
            'postfix': 'spark_model', 'region': region,
            'instance_type': instance_type, 'instance_count': instance_count,
            'deployment_mode': deployment_mode, 'app_name': app_name,
        }
        
        request = requests.post(get_pod_uri('mlflow', '5002') + "/deploy", json=request_payload)
        if request.ok:
            print("Your Job has been submitted. View its status on port 5003 (Job Dashboard)")
            print(request.json)
            return request.json
        else:
            print("Error! An error occured while submitting your job")
            print(request.text)
            return request.text
    
    @check_active
    def deploy_active_run_sagemaker(self, app_name, 
                             region='us-east-2', instance_type='ml.m4.xlarge',
                             instance_count=1, deployment_mode='create'):
        """
        Queue Job to deploy the active run to sagemaker
        
        :param run_id: the id of the run to deploy
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
        return self.deploy_run_sagemaker(self.active_run.info.run_uuid, app_name,
                                        region=region, instance_type=instance_type,
                                        instance_count=instance_count, deployment_mode=deployment_mode)
        
        
