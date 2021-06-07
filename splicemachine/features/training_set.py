from .training_view import TrainingView
from .feature import Feature
from typing import List, Optional
from datetime import datetime
from splicemachine import SpliceMachineException

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
                 create_time: datetime,
                 start_time: Optional[datetime] = None,
                 end_time: Optional[datetime] = None,
                 training_set_id: Optional[int] = None,
                 training_set_version: Optional[int] = None,
                 training_set_name: Optional[str] = None
                 ):
        self.training_view = training_view
        self.features = features
        self.create_time = create_time
        self.training_set_id = training_set_id
        self.training_set_version = training_set_version
        self.training_set_name = training_set_name
        self.start_time = start_time or datetime(year=1900,month=1,day=1) # Saw problems with spark handling datetime.min
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
            try:
                mlflow_ctx.lp("splice.feature_store.training_view_id",self.training_view.view_id)
                mlflow_ctx.lp("splice.feature_store.training_view_version",self.training_view.view_version)
                mlflow_ctx.lp("splice.feature_store.training_set_start_time",str(self.start_time))
                mlflow_ctx.lp("splice.feature_store.training_set_end_time",str(self.end_time))
                mlflow_ctx.lp("splice.feature_store.training_set_create_time",str(self.create_time))
                mlflow_ctx.lp("splice.feature_store.training_set_num_features", len(self.features))
                mlflow_ctx.lp("splice.feature_store.training_set_label", self.training_view.label_column)
                if self.training_set_id and self.training_set_version and self.training_set_name:
                    mlflow_ctx.lp("splice.feature_store.training_set_id",self.training_set_id)
                    mlflow_ctx.lp("splice.feature_store.training_set_version",self.training_set_version)
                    mlflow_ctx.lp("splice.feature_store.training_set_name",self.training_set_name)
                for i,f in enumerate(self.features):
                    mlflow_ctx.lp(f'splice.feature_store.training_set_feature_{i}',f.name)
            except:
                raise SpliceMachineException("It looks like your active run already has a Training Set logged to it. "
                                             "You cannot get a new active Training Set during an active run if you "
                                             "already have an active Training Set. If you've called fs.get_training_set "
                                             "or fs.get_training_set_from_view before starting this run, then that "
                                             "Training Set was logged to the current active run. If you call "
                                             "fs.get_training_set or fs.get_training_set_from_view before starting an "
                                             "mlflow run, all following runs will assume that Training Set to be the "
                                             "active Training Set (until the next call to either of those functions), "
                                             "and will log the Training Set as metadata. For more information, "
                                             "refer to the documentation. If you'd like to use a new Training Set, "
                                             "end the current run, call one of the mentioned functions, and start "
                                             "your new run. Or, call mlflow.remove_active_training_set()") from None
