from splicemachine.mlflow_support import *
from splicemachine.mlflow_support.mlflow_support import _GORILLA_SETTINGS
import gorilla
import mlflow.mleap

def _log_model(model, sample_input, name='mleap_model', **flavor_options):
    flavor_options['sample_input'] = sample_input
    mlflow.log_model(model, name=name, model_lib='mleap', **flavor_options)

gorilla.apply(gorilla.Patch(mlflow.mleap, _log_model.__name__.lstrip('_'), _log_model, settings=_GORILLA_SETTINGS))
