import mlflow
import mlflow.h2o
import mlflow.sklearn
import mlflow.spark
from mlflow.tracking import MlflowClient


def get_pod_uri(pod, port, pod_count=0):
	import os
	try:
		return 'http://{pod}-{pod_count}-node.{framework}.mesos:{port}'.format(pod=pod, pod_count=pod_count, framework=os.environ['FRAMEWORK_NAME'], port=port)
	except KeyError as e:
		raise KeyError("Uh Oh! FRAMEWORK_NAME variable was not found... are you running in Zeppelin?")

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

    @property
    def run():
        """
        Returns the UUID of the current run
        """
        return self.active_run.info.run_uuid

    @property
    def experiment():
        """
        Returns the UUID of the current experiment
        """
        return self.active_experiment.experiment_id

    def __repr__(self):
        return "MLManager: Active Experiment: " + str(self.active_experiment) + \
            " | Active Run: " + str(self.active_run)

    def __str__(self):
        return self.__repr__()

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
        if experiment:
            print("Experiment " + experiment_name + " already exists... setting to active experiment")
            self.active_experiment = experiment
            print("Active experiment has id " + str(experiment.experiment_id))
            if reset:
                print("Keyword argument \"reset\" was set to True. Overwriting experiment and its associated runs...")
                experiment_id = self.active_experiment.experiment_id
                associated_runs = self.list_run_infos(experiment_id)
                for run in associated_runs:
                    print("Deleting run with UUID " + run.run_uuid)
                    manager.delete_run(run.run_uuid)
                print("Successfully overwrote experiment")
        else:
            experiment_id = super(MLManager, self).create_experiment(experiment_name)
            print("Created experiment w/ id=" + str(experiment_id))
            self.set_active_experiment(experiment_id)


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
                "You must set an experiment before you can create a run. Use MLFlowManager.set_active_experiment")

        self.active_run = super(MLManager, self).create_run(self.active_experiment.experiment_id)

        for key in run_metadata:
            self.set_tag(key, run_metadata[key])

    def reset_run(self):
        """
        Reset the current run (deletes logged parameters, metrics, artifacts etc.)
        """
        self.delete_run(self.active_run.info.run_uuid)
        self.create_new_run()

    def set_active_run(self, run_id):
        """
        Set the active run to a previous run (allows you to log metadata for completed run)
        :param run_id: the run UUID for the previous run
        """
        self.active_run = self.get_run(run_id)

    def __log_param(self, *args, **kwargs):
        super(MLManager, self).log_param(self.active_run.info.run_uuid, *args, **kwargs)

    def log_param(self, *args, **kwargs):
        """
        Log a parameter for the active run
        """
        self.__log_param(*args, **kwargs)

    def log_params(self, params):
        """
        Log a list of parameters in order
        :param params: a list of tuples containing parameters mapped to parameter values
        """
        for parameter in params:
            self.log_param(*parameter)

    def __set_tag(self, *args, **kwargs):
        super(MLManager, self).set_tag(self.active_run.info.run_uuid, *args, **kwargs)

    def set_tag(self, *args, **kwargs):
        """
        Set a tag for the active run
        """
        self.__set_tag(*args, **kwargs)

    def set_tags(self, tags):
        """
        Log a list of tags in order
        :param params: a list of tuples containing tags mapped to tag values
        """
        for tag in tags:
            self.set_tag(*tag)

    def __log_metric(self, *args, **kwargs):
        super(MLManager, self).log_metric(self.active_run.info.run_uuid, *args, **kwargs)

    def log_metric(self, *args, **kwargs):
        """
        Log a metric for the active run
        """
        self.__log_metric(*args, **kwargs)

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

    def log_artifacts(self, *args, **kwargs):
        """
        Log artifacts for the active run
        """
        self.__log_artifacts(*args, **kwargs)

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

        with mlflow.start_run(run_uuid=self.active_run.info.run_uuid):
            module.log_model(model, "spark_model")

    def log_spark_model(self, model):
        """
        Log a spark model
        :param model: the fitted pipeline/model to log
        """
        self.log_model(model, mlflow.spark)

    def log_spark_models(self, models):
        """
        Log a list of spark models in order
        :param models: a list of spark models (fitted) to log
        """
        for model in models:
            self.log_spark_model(model)

    def __log_batch(self, *args, **kwargs):
        """
        Log batch metrics, parameters, tags
        """
        super(MLManager, self).log_batch(self.active_run.info.run_uuid, *args, **kwargs)

    def log_batch(self, metrics, parameters, tags):
        """
        Log a batch set of metrics, parameters and tags
        :param metrics: a list of tuples mapping metrics to metric values
        :param parameters: a list of tuples mapping parameters to parameter values
        :param tags: a list of tupples mapping tags to tag values
        """
        self.__log_batch(metrics, parameters, tags)

    @staticmethod
    def _readable_pipeline_stage(pipeline_stage):
        """
        Get a readable version of the Pipeline stage
        (without the memory address)
        :param stage_name: the name of the stage to parse
        """
        if '_' in str(pipeline_stage):
            return str(pipeline_stage).split('_')[0]
        return str(pipeline_stage)

    def log_pipeline_stages(self, fitted_pipeline):
        """
        Log the human-friendly names of each stage in
        a *fitted* Spark pipeline.

        *Warning*: With a big pipeline, this could result in
        a lot of parameters in MLFlow. It is probably best
        to log them yourself, so you can ensure useful tracking

        :param fitted_pipeline: the fitted pipeline object
        """
        for stage_number, pipeline_stage in enumerate(fitted_pipeline.stages):
            readable_stage_name = self._readable_pipeline_stage(pipeline_stage)
            self.log_param('Stage' + str(stage_number), readable_stage_name)

    def log_feature_pipeline_stages(self, fitted_pipeline):
        """
        Log the preprocessing stages applied to each feature in
        the pipeline. Sometimes, this can reduce the amount of output
        but it might be harder to read (as some parameters are long)
        """
        feature_stages = {}

        for stage in fitted_pipeline.stages:
            name = self._readable_pipeline_stage(stage)
            if hasattr(stage, 'getInputCol'): # single-column transformer e.g. StringIndexer
                feature_stages.setdefault(stage.getInputCol(), [])
                feature_stages[stage.getInputCol()].append(name)
            elif hasattr(stage, 'getInputCols'): # multi-column transformer e.g. VectorAssembler
                for inputColumn in stage.getInputCols():
                    feature_stages.setdefault(inputColumn, [])
                    feature_stages[inputColumn].append(name)
            else:
                print("Warning: Cannot parse inputColumns from stage: " + name)
        for column in feature_stages:
            self.log_param(column, ', '.join(feature_stages[column]))

    def terminate(self, create=False, metadata={}):
        """
        Terminate the current run
        """
        self.active_run = None

        if create:
            self.create_new_run(metadata)

    def delete(self):
        """
        Delete the current run
        """
        self.delete_run(self.active_run.info.run_uuid)

