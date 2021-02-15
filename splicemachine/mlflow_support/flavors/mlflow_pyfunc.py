from splicemachine.mlflow_support import *
from splicemachine.mlflow_support.mlflow_support import _GORILLA_SETTINGS
import gorilla
import mlflow.pyfunc

def _log_model(name='pyfunc_model', **flavor_options):
    model = None
    if 'python_model' in flavor_options:
        model = flavor_options.pop('python_model')
    mlflow.log_model(model, name=name, model_lib='pyfunc', **flavor_options)

gorilla.apply(gorilla.Patch(mlflow.pyfunc, _log_model.__name__.lstrip('_'), _log_model, settings=_GORILLA_SETTINGS))
