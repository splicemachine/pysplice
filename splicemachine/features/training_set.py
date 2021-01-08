from .training_view import TrainingView
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
                 training_view: TrainingView,
                 features: List[Feature],
                 start_time: Optional[datetime] = None,
                 end_time: Optional[datetime] = None
                 ):
        self.training_view = training_view
        self.features = features
        self.start_time = start_time or datetime(year=1990,month=1,day=1) # Saw problems with spark handling datetime.min
        self.end_time = end_time or datetime.today()

    def _register_metadata(self, mlflow_ctx):
        """
        Registers training set with mlflow if the user has registered the feature store in their mlflow session,
        and has called either get_training_set or get_training_set_from_view before or during an mlflow run

        :param mlflow_ctx: the mlflow context
        :return: None
        """
        if mlflow_ctx.active_run():
            print("There is an active mlflow run, your training set will be logged to that run.")
            mlflow_ctx.lp("splice.feature_store.training_set",self.training_view.name)
            mlflow_ctx.lp("splice.feature_store.training_set_start_time",str(self.start_time))
            mlflow_ctx.lp("splice.feature_store.training_set_end_time",str(self.end_time))
            mlflow_ctx.lp("splice.feature_store.training_set_num_features", len(self.features))
            for i,f in enumerate(self.features):
                mlflow_ctx.lp(f'splice.feature_store.training_set_feature_{i}',f.name)
