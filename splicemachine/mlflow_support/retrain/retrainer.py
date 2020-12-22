from abc import ABC, abstractmethod
from cron_descriptor import get_description, FormatException

class Retrainer(ABC):
    """
    Splice Machine class for model retraining
    """
    def __init__(self, cron_exp, run_id, conda_env: str=None, _create_contexts=False):
        """
        Initialize Global Contexts for Retraining
        """
        self.cron_exp = cron_exp
        self.run_id = run_id
        self.conda_env = conda_env
        self.__check_cron()

    def __check_cron(self):
        try:
            print(f"You've created a retrainer scheduled for {get_description(self.cron_exp)}")
        except FormatException:
            raise Exception(f'The provided cron "{self.cron_exp}" is invalid. '
                                         f'See above for more information')

    
    @property
    def has_conda(self):
        """
        Check whether or not the user specified a conda.yaml
        """
        return bool(self.conda_env)
    
    def _create_contexts(self):
        """
        Internal method to create contexts for retraining
        """
        self.create_spark()
        self.create_splice()
        self.create_feature_store()
        self.create_mlflow()
        
    def create_spark(self):
        """
        Create a SparkSession for Retraining. The Sessions's creation can
        be overriden by subclassing the `Retrainer` and redefining the function.
        """
        from pyspark.sql import SparkSession
        global spark 
        spark = SparkSession.builder.getOrCreate()
        self.spark = spark
    
    def create_mlflow(self):
        """
        Create an mlflow context for retraining. Its creation
        can be overriden by subclassing the `Retrainer` and redefining the function
        """
        from splicemachine.mlflow_support import mlflow_support
        mlflow_support.main()
        global mlflow
        mlflow = mlflow_support.mlflow
        mlflow.register_splice_context(splice)
        mlflow.register_feature_store(fs)
        self.mlflow = mlflow
    
    def create_splice(self):
        """
        Create a PySpliceContext for retraining. Its creation can be 
        overriden by subclassing the `Retrainer` and redefining the function,
        which is required if you would like to establish a connection to a external
        Splice cluster using the ExtPySpliceContext.
        """
        from splicemachine.spark import PySpliceContext
        global splice
        splice = PySpliceContext(spark)
        self.splice = splice
    
    def create_feature_store(self):
        """
        Create a FeatureStore object for retraining. Its creation
        can be overriden by subclassing the `Retrainer` and redefining the function
        """
        pass
        from splicemachine.features import FeatureStore
        global fs
        fs = FeatureStore(splice)
        self.fs = fs
    
    @abstractmethod
    def retrain(self):
        """
        Function containing the logic for retraining. Because we leverage
        cloudpickle, this function can reference other functions/classes that
        are defined in the notebook (without having to redefine them in your class).
        Global contexts, "splice" (PySpliceContext), "spark" (SparkSession), "mlflow" (mlflow), and "fs" (feature store)
        Also, to create nested runs for all retrains, use mlflow.start_nested_run(run_id=self.run_id) 
        instead of mlflow.start_run().
        """
        pass
