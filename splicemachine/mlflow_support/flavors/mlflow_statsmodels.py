from splicemachine.mlflow_support import *
from splicemachine.mlflow_support.mlflow_support import _GORILLA_SETTINGS
import gorilla
import mlflow.statsmodels

def _log_model(model, name='statsmodels_model', **flavor_options):
    mlflow.log_model(model, name=name, model_lib='statsmodels', **flavor_options)

gorilla.apply(gorilla.Patch(mlflow.statsmodels, _log_model.__name__.lstrip('_'), _log_model, settings=_GORILLA_SETTINGS))
