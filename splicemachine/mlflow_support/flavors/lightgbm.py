from splicemachine.mlflow_support import *
from splicemachine.mlflow_support.mlflow_support import _GORILLA_SETTINGS
import gorilla
import mlflow.lightgbm

def _log_model(model, name=None, **flavor_options):
    mlflow.log_model(model, name=name, model_lib='lightgbm', **flavor_options)

gorilla.apply(gorilla.Patch(mlflow.lightgbm, _log_model.__name__.lstrip('_'), _log_model, settings=_GORILLA_SETTINGS))
