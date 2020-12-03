from .training_context import TrainingContext
from .feature import Feature
from typing import List, Optional
from datetime import datetime

class TrainingSet:
    """
    A training set is a training context with a specific set of desired features, a start time for the history of
    those features and an end time. This amounts to a historical bath of features used for training models.
    A training set is defined by the start and end time, so using the same training context over a different time window
    amounts to a different training_set
    """
    def __init__(self,
                 *,
                 training_context: TrainingContext,
                 features: List[Feature],
                 start_time: Optional[datetime],
                 end_time: Optional[datetime]
                 ):
        self.training_context = training_context
        self.features = features
        self.start_time = start_time or datetime.min
        self.end_time = end_time or datetime.today()

    def _register_metadata(self, mlflow_ctx):
        if mlflow_ctx.active_run():
            print("There is an active mlflow run, your training set will be logged to that run.")
            mlflow_ctx.lp("splice.feature_store.training_set",self.training_context.name)
            mlflow_ctx.lp("splice.feature_store.training_set_start_time",str(self.start_time))
            mlflow_ctx.lp("splice.feature_store.training_set_end_time",str(self.end_time))
            mlflow_ctx.lp("splice.feature_store.training_set_num_features", len(self.features))
            for i,f in enumerate(self.features):
                mlflow_ctx.lp(f'splice.feature_store.training_set_feature_{i}',f.name)
