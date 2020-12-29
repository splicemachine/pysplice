import time
from abc import ABC, abstractmethod


class AbstractRetrainer(ABC):
    """
    Splice Machine class for model retraining
    """
    CRON_EXP = None
    PARENT_RUN_ID = None

    # , cron_exp, run_id, conda_env: str = None
    def __init__(self):
        """
        Initialize Global Contexts for Retraining
        """
        # Assigned when contexts are created
        self.spark = None
        self.splice = None
        self.mlflow = None
        self.fs = None

        self._create_contexts()

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
        from splicemachine.features import FeatureStore
        global fs
        fs = FeatureStore(splice)
        self.fs = fs

    @staticmethod
    def test(run_id: str = None):
        """
        Test the logic of your retraining (bypassing cron schedule)
        :param run_id: parent run id for testing purposes
        """
        retrainer = Retrainer()
        Retrainer.PARENT_RUN_ID = run_id
        start_time = time.time()
        retrainer.retrain()
        end_time = time.time()
        print(f"Retrainer completed execution in {(end_time - start_time) * 1000} ms")
        retrainer.PARENT_RUN_ID = None
        del retrainer

    @abstractmethod
    def retrain(self):
        """
        Function containing the logic for retraining. Because we leverage
        cloudpickle, this function can reference other functions/variables/classes that
        are defined in the notebook (without having to redefine them in your class).
        Global contexts, "splice" (PySpliceContext), "spark" (SparkSession), "mlflow" (mlflow), and "fs" (feature store)
        Also, to create nested runs for all retrains, use mlflow.start_nested_run(run_id=self.run_id) 
        instead of mlflow.start_run().
        """
        pass
