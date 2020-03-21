import time
from functools import partial as fix_params
from collections import defaultdict
from contextlib import contextmanager
from io import BytesIO
from zipfile import ZipFile

import gorilla
import mlflow
import path
from py4j.java_gateway import java_import

from splicemachine.mlflow_support.utilities import *

_TESTING = env_vars.get("TESTING", False)
_TRACKING_URL = get_pod_uri("mlflow", "5001", _TESTING)

_CLIENT = mlflow.tracking.MlflowClient(tracking_uri=_TRACKING_URL)

_GORILLA_SETTINGS = gorilla.Settings(allow_hit=True, store_hit=True)


def _mlflow_patch(name):
    """
    Create a MLFlow Patch that applies the
    default gorilla settings
    :param name: destination name under mlflow package
    :return: decorator for patched function
    """
    return gorilla.patch(mlflow, name, settings=_GORILLA_SETTINGS)


def _get_current_run_data():
    """
    Get the data associated with the current run.
    As of MLFLow 1.6, it currently does not support getting

    run info from the mlflow.active_run object, so we need it
    to be retrieved via the tracking client.
    :return: active run data object
    """
    return _CLIENT.get_run(mlflow.active_run().info.run_id).data


@_mlflow_patch('register_splice_context')
def _register_splice_context(splice_context):
    """
    Register a Splice Context for Spark/Database operations
    (artifact storage, for example)
    :param splice_context:  splice context to input
    """
    mlflow._splice_context = splice_context


def _check_for_splice_ctx(func):
    """
    Check to make sure that the user has registered
    a PySpliceContext with the mlflow object before allowing
    spark operations to take place
    :param func: function to wrap before checking
    """

    def wrapped(*args, **kwargs):
        if not mlflow._splice_context:
            raise SpliceMachineException(
                "You must run `mlflow.register_splice_context(py_splice_context) before "
                "you can run mlflow artifact operations!"
            )
        return func(*args, **kwargs)

    return wrapped


@_mlflow_patch('current_run_id')
def _current_run_id():
    """
    Retrieve the current run id
    :return: the current run id
    """
    return mlflow.active_run().info.run_uuid


@_mlflow_patch('current_exp_id')
def _current_exp_id():
    """
    Retrieve the current exp id
    :return: the current experiment id
    """
    return mlflow.active_run().info.experiment_id


@_mlflow_patch('lp')
def _lp(key, value):
    """
    Add a shortcut for logging parameters in MLFlow.
    Accessible from mlflow.lp
    :param key: key for the parameter
    :param value: value for the parameter
    """
    mlflow.log_param(key, value)


@_mlflow_patch('lm')
def _lm(key, value):
    """
    Add a shortcut for logging metrics in MLFlow.
    Accessible from mlflow.lm
    :param key: key for the parameter
    :param value: value for the parameter
    """
    mlflow.log_metric(key, value)


@_mlflow_patch('log_spark_model')
@_check_for_splice_ctx
def _log_spark_model(model, name='model'):
    """
    Log a fitted spark pipeline or model
    :param model: (PipelineModel or Model) is the fitted Spark Model/Pipeline to store
        with the current run
    :param name: (str) the run relative name to store the model under
    """
    if _get_current_run_data().tags['splice.model_name']:  # this function has already run
        raise Exception("Only one model is permitted per run.")

    mlflow.set_tag('splice.model_name', name)  # read in backend for deployment

    jvm = mlflow._splice_context.jvm
    java_import(jvm, "java.io.{BinaryOutputStream, ObjectOutputStream, ByteArrayInputStream}")

    if not SparkUtils.is_spark_pipeline(model):
        model = PipelineModel(
            stages=[model]
        )  # create a pipeline with only the model if a model is passed in

    baos = jvm.java.io.ByteArrayOutputStream()  # serialize the PipelineModel to a byte array
    oos = jvm.java.io.ObjectOutputStream(baos)
    oos.writeObject(model._to_java())
    oos.flush()
    oos.close()
    insert_artifact(name, baos.toByteArray(), mlflow._splice_context, mlflow.active_run().info.run_uuid,
                    file_ext='sparkmodel')  # write the byte stream to the db as a BLOB


@_mlflow_patch('start_run')
def _start_run(run_id=None, tags=None, experiment_id=None, run_name=None, nested=False):
    """
    Start a new run
    :param tags: a dictionary containing metadata about the current run.
        For example:
            {
                'team': 'pd',
                'purpose': 'r&d'
            }
    :param run_name: an optional name for the run to show up in the MLFlow UI
    :param run_id: if you want to reincarnate an existing run, pass in the run id
    :param experiment_id: if you would like to create an experiment/use one for this run
    :param nested: Controls whether run is nested in parent run. True creates a nest run
    """
    if not tags:
        tags = {}
    tags['mlflow.user'] = get_user()

    for key in tags:
        mlflow.set_tag(key, tags[key])
    orig = gorilla.get_original_attribute(mlflow, "start_run")
    orig(run_id=run_id, experiment_id=experiment_id, run_name=run_name, nested=nested)


@_mlflow_patch('log_pipeline_stages')
def _log_pipeline_stages(pipeline):
    for stage_number, pipeline_stage in enumerate(SparkUtils.get_stages(pipeline)):
        readable_stage_name = SparkUtils.readable_pipeline_stage(pipeline_stage)
        mlflow.log_param('Stage' + str(stage_number), readable_stage_name)


@_mlflow_patch('log_feature_transformations')
def _log_feature_transformations(unfit_pipeline):
    """
    Log feature transformations for an unfit pipeline
    Logs --> feature movement through the pipelien
    :param unfit_pipeline: unfit pipeline to log
    """
    transformations = defaultdict(lambda: [[], None])  # transformations, outputColumn

    for stage in SparkUtils.get_stages(unfit_pipeline):
        input_cols, output_col = SparkUtils.get_cols(stage, get_input=True), SparkUtils.get_cols(stage, get_input=False)
        if input_cols and output_col:  # make sure it could parse transformer
            for column in input_cols:
                first_column_found = find_inputs_by_output(transformations, column)
                if first_column_found:  # column is not original
                    for f in first_column_found:
                        transformations[f][1] = output_col
                        transformations[f][0].append(
                            SparkUtils.readable_pipeline_stage(stage))
                else:
                    transformations[column][1] = output_col
                    transformations[column][0].append(SparkUtils.readable_pipeline_stage(stage))

    for column in transformations:
        param_value = ' -> '.join([column] + transformations[column][0] +
                                  [transformations[column][1]])
        mlflow.log_param('Column- ' + column, param_value)


@_mlflow_patch('log_model_params')
def _log_model_params(pipeline_or_model):
    """
    Log the parameters of a fitted model or a
    model part of a fitted pipeline
    :param pipeline_or_model: fitted pipeline/fitted model
    """
    model = SparkUtils.get_model_stage(pipeline_or_model)

    mlflow.log_param('model', SparkUtils.readable_pipeline_stage(model))
    if hasattr(model, '_java_obj'):
        verbose_parameters = SparkUtils.parse_string_parameters(model._java_obj.extractParamMap())
    elif hasattr(model, 'getClassifier'):
        verbose_parameters = SparkUtils.parse_string_parameters(
            model.getClassifier()._java_obj.extractParamMap())
    else:
        raise Exception("Could not parse model type: " + str(model))
    for param in verbose_parameters:
        try:
            value = float(verbose_parameters[param])
            mlflow.log_param('Hyperparameter- ' + param.split('-')[0], value)
        except:
            mlflow.log_param('Hyperparameter- ' + param.split('-')[0], verbose_parameters[param])


@_mlflow_patch('timer')
@contextmanager
def _timer(timer_name, param=True):
    """
    Context manager for logging
    :param timer_name:
    :param param: whether or not to log the timer as a param (default=True). If false, logs as metric.
    :return:
    """
    try:
        t0 = time.time()
    finally:
        t1 = time.time() - t0
        # Syntactic Sugar
        (mlflow.log_param if param else mlflow.log_metric)(timer_name, t1)


@_mlflow_patch('download_artifact')
@_check_for_splice_ctx
def _download_artifact(name, local_path, run_id=None):
    """
    Download the artifact at the given
    run id (active default) + name
    to the local path
    :param name: (str) artifact name to load
      (with respect to the run)
    :param local_path: (str) local path to download the
      model to. This path MUST include the file extension
    :param run_id: (str) the run id to download the artifact
      from. Defaults to active run
    """
    file_ext = path.splitext(local_path)[1]
    if not file_ext:
        raise ValueError('local_path variable must contain the file extension!')

    run_id = run_id or mlflow.active_run().info.run_uuid
    blob_data = SparkUtils.retrieve_artifact_stream(mlflow._splice_context, run_id, name)
    if file_ext == '.zip':
        zip_file = ZipFile(BytesIO(blob_data))
        zip_file.extractall()
    else:
        with open(local_path, 'wb') as artifact_file:
            artifact_file.write(blob_data)


@_mlflow_patch('load_spark_model')
@_check_for_splice_ctx
def _load_spark_model(run_id=None, name='model'):
    """
    Download a model from database
    and load it into Spark
    :param run_id: the id of the run to get a model from
        (the run must have an associated model with it named spark_model)
    :param name: the name of the model in the database
    """
    run_id = run_id or mlflow.active_run().info.run_uuid
    spark_pipeline_blob = SparkUtils.retrieve_artifact_stream(mlflow._splice_context, run_id, name)
    bis = mlflow._splice_context.jvm.java.io.ByteArrayInputStream(spark_pipeline_blob)
    ois = mlflow._splice_context.jvm.java.io.ObjectInputStream(bis)
    pipeline = PipelineModel._from_java(ois.readObject())  # convert object from Java
    # PipelineModel to Python PipelineModel
    ois.close()

    if len(pipeline.stages) == 1 and SparkUtils.is_spark_pipeline(pipeline.stages[0]):
        pipeline = pipeline.stages[0]

    return pipeline


@_mlflow_patch('log_artifact')
@_check_for_splice_ctx
def _log_artifact(file_name, name, run_uuid=None):
    """
    Log an artifact for the active run
    :param file_name: (str) the name of the file name to log
    :param name: (str) the name of the run relative name to store the model under
    :param run_uuid: the run uuid of a previous run, if none, defaults to current run
    NOTE: We do not currently support logging directories. If you would like to log a directory, please zip it first
          and log the zip file
    """
    file_ext = path.splitext(file_name)[1].lstrip('.')
    with open(file_name, 'rb') as artifact:
        byte_stream = bytearray(bytes(artifact.read()))

    run_id = run_uuid if run_uuid else mlflow.active_run().info.run_uuid

    insert_artifact(mlflow._splice_context, name, byte_stream, run_id, file_ext=file_ext)


def apply_patches():
    """
    Apply all the Gorilla Patches;
    ALL GORILLA PATCHES SHOULD BE PREFIXED WITH "_" BEFORE THEIR DESTINATION IN MLFLOW
    """
    targets = [_register_splice_context, _lp, _lm, _timer, _log_artifact, _log_feature_transformations,
               _log_model_params, _log_pipeline_stages, _log_spark_model, _load_spark_model, _download_artifact,
               _start_run, _current_run_id, _current_exp_id]

    for target in targets:
        gorilla.apply(gorilla.Patch(mlflow, target.__name__.lstrip('_'), target, settings=_GORILLA_SETTINGS))


def main():
    mlflow.set_tracking_uri(_TRACKING_URL)
    apply_patches()


main()
