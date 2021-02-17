from splicemachine.mlflow_support import *
from splicemachine.mlflow_support.mlflow_support import _GORILLA_SETTINGS
import gorilla
import mlflow.pytorch

def _log_model(model, name='pytorch_model', **flavor_options):
    mlflow.log_model(model, name=name, model_lib='pytorch', **flavor_options)

gorilla.apply(gorilla.Patch(mlflow.pytorch, _log_model.__name__.lstrip('_'), _log_model, settings=_GORILLA_SETTINGS))
