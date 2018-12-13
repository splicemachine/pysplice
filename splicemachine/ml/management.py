import mlflow
import mlflow.h2o
import mlflow.sklearn
import mlflow.spark
from mlflow.tracking import MlflowClient


class MLManager(MlflowClient):
    """
    A class for managing your MLFlow Runs/Experiments
    """

    def __init__(self, _tracking_uri='http://mlflow:5001'):
        mlflow.set_tracking_uri(_tracking_uri)
        print("Tracking Model Metadata on MLFlow Server @ " + mlflow.get_tracking_uri())

        if not mlflow.get_tracking_uri() == _tracking_uri:
            Warning("MLManager doesn't seem to be communicating with the right server endpoint."
                    "Try instantiating this class again!")

        MlflowClient.__init__(self, _tracking_uri)
        self.active_run = None
        self.active_experiment = None

    @staticmethod
    def __removekey(d, key):
        """
        Remove a key from a dictionary
        """
        r = dict(d)
        del r[key]
        return r

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

    def create_new_run(self, user_id="splice"):
        """
        Create a new run in the active experiment and set it to be active
        :param user_id: the user who creates the run in the MLFlow UI
        """
        if not self.active_experiment:
            raise Exception(
                "You must set an experiment before you can create a run. Use MLFlowManager.set_active_experiment")

        self.active_run = self.create_run(self.active_experiment.experiment_id, user_id=user_id)

    def set_active_run(self, run_id):
        """
        Set the active run to a previous run (allows you to log metadata for completed run)
        :param run_id: the run UUID for the previous run 
        """
        self.active_run = self.get_run(run_id)
        
    def get_experiment_by_name(self, experiment_name):
        super(MLManager, self).get_experiment_by_name(experiment_name)
      
    def get_experiment(self, experiment_name):
        super(MLManager, self).get_experiment(experiment_name)
        
    def get_run(self, run_id):
        super(MLManager, self).get_run(run_id)

    def __log_param(self, *args, **kwargs):
        super(MLManager, self).log_param(self.active_run.info.run_uuid, *args, **kwargs)

    def log_param(self, *args, **kwargs):
        """
        Log a parameter for the active run
        """
        self.__log_param(*args, **kwargs)

    def __set_tag(self, *args, **kwargs):
        super(MLManager, self).set_tag(self.active_run.info.run_uuid, *args, **kwargs)

    def set_tag(self, *args, **kwargs):
        """
        Set a tag for the active run
        """
        self.__set_tag(*args, **kwargs)

    def __log_metric(self, *args, **kwargs):
        super(MLManager, self).log_metric(self.active_run.info.run_uuid, *args, **kwargs)

    def log_metric(self, *args, **kwargs):
        """
        Log a metric for the active run
        """
        self.__log_metric(*args, **kwargs)

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
