import time
from collections import defaultdict
from contextlib import contextmanager
from io import BytesIO
from os import path
from zipfile import ZipFile

import gorilla
import mlflow
import requests
from requests.auth import HTTPBasicAuth
from mleap.pyspark import spark_support

from splicemachine.mlflow_support.utilities import *
from splicemachine.spark.context import PySpliceContext
from splicemachine.spark.constants import CONVERSIONS

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
    assert isinstance(splice_context, PySpliceContext), "You must pass in a PySpliceContext to this method"
    mlflow._splice_context = splice_context


def _check_for_splice_ctx():
    """
    Check to make sure that the user has registered
    a PySpliceContext with the mlflow object before allowing
    spark operations to take place
    """

    if not getattr(mlflow, '_splice_context'):
        raise SpliceMachineException(
            "You must run `mlflow.register_splice_context(py_splice_context) before "
            "you can run this mlflow operation!"
        )


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
def _log_spark_model(model, name='model'):
    """
    Log a fitted spark pipeline or model
    :param model: (PipelineModel or Model) is the fitted Spark Model/Pipeline to store
        with the current run
    :param name: (str) the run relative name to store the model under
    """
    _check_for_splice_ctx()
    if _get_current_run_data().tags.get('splice.model_name'):  # this function has already run
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
    insert_artifact(mlflow._splice_context, name, baos.toByteArray(), mlflow.active_run().info.run_uuid,
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
    # Get the current running transaction ID for time travel/data governance
    _check_for_splice_ctx()
    db_connection = mlflow._splice_context.getConnection()
    prepared_statement = db_connection.prepareStatement('CALL SYSCS_UTIL.SYSCS_GET_CURRENT_TRANSACTION()')
    x = prepared_statement.executeQuery()
    x.next()
    timestamp = x.getInt(1)
    prepared_statement.close()

    tags = tags if tags else {}
    tags['mlflow.user'] = get_user()
    tags['DB_Transaction_ID'] = timestamp

    orig = gorilla.get_original_attribute(mlflow, "start_run")
    active_run = orig(run_id=run_id, experiment_id=experiment_id, run_name=run_name, nested=nested)

    for key in tags:
        mlflow.set_tag(key, tags[key])
    if not run_id:
        mlflow.set_tag('run_id',mlflow._current_run_id())
    if run_name:
        mlflow.set_tag('mlflow.runName', run_name)

    return active_run


@_mlflow_patch('log_pipeline_stages')
def _log_pipeline_stages(pipeline):
    """
    Log the pipeline stages as params for the run
    :param pipeline: fitted/unitted pipeline
    """
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
        yield
    finally:
        t1 = time.time() - t0
        # Syntactic Sugar
        (mlflow.log_param if param else mlflow.log_metric)(timer_name, t1)
        print(
            f"Code Block {timer_name}:\nRan in {round(t1, 3)} secs\nRan in {round(t1 / 60, 3)} mins"
        )


@_mlflow_patch('download_artifact')
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
    _check_for_splice_ctx()
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
def _load_spark_model(run_id=None, name='model'):
    """
    Download a model from database
    and load it into Spark
    :param run_id: the id of the run to get a model from
        (the run must have an associated model with it named spark_model)
    :param name: the name of the model in the database
    """
    _check_for_splice_ctx()
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
def _log_artifact(file_name, name, run_uuid=None):
    """
    Log an artifact for the active run
    :param file_name: (str) the name of the file name to log
    :param name: (str) the name of the run relative name to store the model under
    :param run_uuid: the run uuid of a previous run, if none, defaults to current run
    NOTE: We do not currently support logging directories. If you would like to log a directory, please zip it first
          and log the zip file
    """
    _check_for_splice_ctx()
    file_ext = path.splitext(file_name)[1].lstrip('.')

    with open(file_name, 'rb') as artifact:
        byte_stream = bytearray(bytes(artifact.read()))

    run_id = run_uuid if run_uuid else mlflow.active_run().info.run_uuid

    insert_artifact(mlflow._splice_context, name, byte_stream, run_id, file_ext=file_ext)


@_mlflow_patch('login_director')
def _login_director(username, password):
    """
    Authenticate into the MLManager Director
    :param username: database username
    :param password: database password
    """
    mlflow._basic_auth = HTTPBasicAuth(username, password)


def _initiate_job(payload, endpoint):
    """
    Send a job to the initiation endpoint
    :param payload: (dict) JSON payload for POST request
    :param endpoint: (str) REST endpoint to target
    :return: (str) Response text from request
    """
    if not getattr(mlflow, '_basic_auth'):
        raise Exception(
            "You have not logged into MLManager director."
            " Please run mlflow.login_director(username, password)"
        )
    request = requests.post(
        get_pod_uri('mlflow', 5003, _testing=_TESTING) + endpoint,
        auth=mlflow._basic_auth,
        json=payload,

    )

    if request.ok:
        print("Your Job has been submitted. View its status on port 5003 (Job Dashboard)")
        print(request.json)
        return request.json
    else:
        print("Error! An error occurred while submitting your job")
        print(request.text)
        return request.text


@_mlflow_patch('deploy_aws')
def _deploy_aws(app_name, region='us-east-2', instance_type='ml.m5.xlarge',
                run_id=None, instance_count=1, deployment_mode='replace'):
    """
    Queue Job to deploy a run to sagemaker with the
    given run id (found in MLFlow UI or through search API)
    :param run_id: the id of the run to deploy. Will default to the current
        run id.
    :param app_name: the name of the app in sagemaker once deployed
    :param region: the sagemaker region to deploy to (us-east-2,
        us-west-1, us-west-2, eu-central-1 supported)
    :param instance_type: the EC2 Sagemaker instance type to deploy on
        (ml.m4.xlarge supported)
    :param instance_count: the number of instances to load balance predictions
        on
    :param deployment_mode: the method to deploy; create=application will fail
        if an app with the name specified already exists; replace=application
        in sagemaker will be replaced with this one if app already exists;
        add=add the specified model to a prexisting application (not recommended)
    """
    # get run from mlflow
    print("Processing...")
    time.sleep(3)  # give the mlflow server time to register the artifact, if necessary

    supported_aws_regions = ['us-east-2', 'us-west-1', 'us-west-2', 'eu-central-1']
    supported_instance_types = ['ml.m5.xlarge']
    supported_deployment_modes = ['replace', 'add']

    # data validation
    if region not in supported_aws_regions:
        raise Exception("Region must be in list: " + str(supported_aws_regions))
    if instance_type not in supported_instance_types:
        raise Exception("Instance type must be in list: " + str(instance_type))
    if deployment_mode not in supported_deployment_modes:
        raise Exception("Deployment mode must be in list: " + str(supported_deployment_modes))

    request_payload = {
        'handler_name': 'DEPLOY_AWS', 'run_id': run_id if run_id else mlflow.active_run().info.run_uuid,
        'region': region, 'user': get_user(),
        'instance_type': instance_type, 'instance_count': instance_count,
        'deployment_mode': deployment_mode, 'app_name': app_name
    }

    return _initiate_job(request_payload, '/api/rest/initiate')


@_mlflow_patch('deploy_azure')
def _deploy_azure(endpoint_name, resource_group, workspace, run_id=None, region='East US',
                  cpu_cores=0.1, allocated_ram=0.5, model_name=None):
    """
    Deploy a given run to AzureML.
    :param endpoint_name: (str) the name of the endpoint in AzureML when deployed to
        Azure Container Services. Must be unique.
    :param resource_group: (str) Azure Resource Group for model. Automatically created if
        it doesn't exist.
    :param workspace: (str) the AzureML workspace to deploy the model under.
        Will be created if it doesn't exist
    :param run_id: (str) if specified, will deploy a previous run (
        must have an spark model logged). Otherwise, will default to the active run
    :param region: (str) AzureML Region to deploy to: Can be East US, East US 2, Central US,
        West US 2, North Europe, West Europe or Japan East
    :param cpu_cores: (float) Number of CPU Cores to allocate to the instance.
        Can be fractional. Default=0.1
    :param allocated_ram: (float) amount of RAM, in GB, allocated to the container.
        Default=0.5
    :param model_name: (str) If specified, this will be the name of the model in AzureML.
        Otherwise, the model name will be randomly generated.
    """
    supported_regions = ['East US', 'East US 2', 'Central US',
                         'West US 2', 'North Europe', 'West Europe', 'Japan East']

    if region not in supported_regions:
        raise Exception("Region must be in list: " + str(supported_regions))
    if cpu_cores <= 0:
        raise Exception("Invalid CPU Count")
    if allocated_ram <= 0:
        raise Exception("Invalid Allocated RAM")

    request_payload = {
        'handler_name': 'DEPLOY_AZURE',
        'endpoint_name': endpoint_name,
        'resource_group': resource_group,
        'workspace': workspace,
        'run_id': run_id if run_id else mlflow.active_run().info.run_uuid,
        'cpu_cores': cpu_cores,
        'allocated_ram': allocated_ram,
        'model_name': model_name
    }
    return _initiate_job(request_payload, '/api/rest/initiate')


@_mlflow_patch('deploy_database')
def _deploy_db(fittedPipe, df, db_schema_name, db_table_name, primary_key,
               run_id=None, classes=None, verbose=False, replace=False) -> None:
    """
    Function to deploy a trained (Spark for now) model to the Database. This creates 2 tables: One with the features of the model, and one with the prediction and metadata.
    They are linked with a column called MOMENT_ID

    :param fittedPipe: (spark pipeline or model) The fitted pipeline to deploy
    :param df: (Spark DF) The dataframe used to train the model
                NOTE: this dataframe should NOT be transformed by the model. The columns in this df are the ones
                that will be used to create the table.
    :param db_schema_name: (str) the schema name to deploy to. If None, the currently set schema will be used.
    :param db_table_name: (str) the table name to deploy to. If none, the run_id will be used for the table name(s)
    :param primary_key: (List[Tuple[str, str]]) List of column + SQL datatype to use for the primary/composite key
    :param run_id: (str) The active run_id
    :param classes: List[str] The classes (prediction values) for the model being deployed.
                    NOTE: If not supplied, the table will have column named c0,c1,c2 etc for each class
    :param verbose: bool Whether or not to print out the queries being created. Helpful for debugging

    This function creates the following:
    * Table (default called DATA_{run_id}) where run_id is the run_id of the mlflow run associated to that model. This will have a column for each feature in the feature vector as well as a MOMENT_ID as primary key
    * Table (default called DATA_{run_id}_PREDS) That will have the columns:
        USER which is the current user who made the request
        EVAL_TIME which is the CURRENT_TIMESTAMP
        MOMENT_ID same as the DATA table to link predictions to rows in the table
        PREDICTION. The prediction of the model. If the :classes: param is not filled in, this will be c0,c1,c2 etc for classification models
        A column for each class of the predictor with the value being the probability/confidence of the model if applicable
    * A trigger that runs on (after) insertion to the data table that runs an INSERT into the prediction table,
        calling the PREDICT function, passing in the row of data as well as the schema of the dataset, and the run_id of the model to run
    * A trigger that runs on (after) insertion to the prediction table that calls an UPDATE to the row inserted, parsing the prediction probabilities and filling in proper column values

    """
    _check_for_splice_ctx()
    classes = classes if classes else []
    run_id = run_id if run_id else mlflow.active_run().info.run_uuid
    db_table_name = db_table_name if db_table_name else f'data_{run_id}'
    schema_table_name = f'{db_schema_name}.{db_table_name}' if db_schema_name else db_table_name

    # Get the VectorAssembler so we can get the features of the model
    # FIXME: this might not be correct. If transformations are made before hitting the VectorAssembler, they
    # FIXME: Also need to be included in the columns of the table. We need the df columns + VectorAssembler inputCols
    # FIXME: We can do something similar to the log_feature_transformations function to get necessary columns
    # FIXME: Or, this may just be df.columns ...
    feature_columns = df.columns
    # Get the datatype of each column in the dataframe
    schema_types = {str(i.name): re.sub("[0-9,()]", "", str(i.dataType)) for i in df.schema}

    # Make sure primary_key is valid format
    validate_primary_key(primary_key)

    # Get model type
    modelType = SparkUtils.get_model_type(fittedPipe)

    print(f'Deploying model {run_id} to table {schema_table_name}')

    if classes:
        if modelType not in (SparkModelType.CLASSIFICATION, SparkModelType.CLUSTERING_WITH_PROB):
            print('Prediction labels found but model is not type Classification. Removing labels')
            classes = None
        else:
            # handling spaces in class names
            classes = [c.replace(' ', '_') for c in classes]
            print(
                f'Prediction labels found. Using {classes} as labels for predictions {list(range(0, len(classes)))} respectively')
    else:
        if modelType in (SparkModelType.CLASSIFICATION, SparkModelType.CLUSTERING_WITH_PROB):
            # Add a column for each class of the prediction to output the probability of the prediction
            classes = [f'C{i}' for i in range(SparkUtils.get_num_classes(fittedPipe))]

    # See if the df passed in has already been transformed.
    # If not, transform it
    if 'prediction' not in df.columns:
        df = fittedPipe.transform(df)
    # Get the Mleap model and insert it into the MODELS table
    mleap_model = get_mleap_model(mlflow._splice_context, fittedPipe, df, run_id)
    insert_mleap_model(mlflow._splice_context, run_id, mleap_model)

    # Create the schema of the table (we use this a few times)
    schema_str = ''
    for i in feature_columns:
        schema_str += f'\t{i} {CONVERSIONS[schema_types[str(i)]]},\n'

    try:
        # Create table 1: DATA
        print('Creating data table ...', end=' ')
        create_data_table(mlflow._splice_context, schema_table_name, schema_str, primary_key, verbose)
        print('Done.')

        # Create table 2: DATA_PREDS
        print('Creating prediction table ...', end=' ')
        create_data_preds_table(mlflow._splice_context, schema_table_name, classes, primary_key, modelType, verbose)
        print('Done.')

        # Create Trigger 1: (model prediction)
        print('Creating model prediction trigger ...', end=' ')
        create_prediction_trigger(mlflow._splice_context, schema_table_name, run_id, feature_columns, schema_types,
                                  schema_str,
                                  primary_key, modelType, verbose)
        print('Done.')

        if modelType in (SparkModelType.CLASSIFICATION, SparkModelType.CLUSTERING_WITH_PROB):
            # Create Trigger 2: model parsing
            print('Creating parsing trigger ...', end=' ')
            create_parsing_trigger(mlflow._splice_context, schema_table_name, primary_key, run_id, classes, verbose)
            print('Done.')
    except Exception as e:
        import traceback
        print('Model deployment failed. Dropping all tables.')
        drop_tables_on_failure(mlflow._splice_context, schema_table_name, run_id)
        if not verbose:
            print('For more insight into the SQL statement that generated this error, rerun with verbose=True')
        traceback.print_exc()

    print('Model Deployed.')


def apply_patches():
    """
    Apply all the Gorilla Patches;
    ALL GORILLA PATCHES SHOULD BE PREFIXED WITH "_" BEFORE THEIR DESTINATION IN MLFLOW
    """
    targets = [_register_splice_context, _lp, _lm, _timer, _log_artifact, _log_feature_transformations,
               _log_model_params, _log_pipeline_stages, _log_spark_model, _load_spark_model, _download_artifact,
               _start_run, _current_run_id, _current_exp_id, _deploy_aws, _deploy_azure, _deploy_db, _login_director]

    for target in targets:
        gorilla.apply(gorilla.Patch(mlflow, target.__name__.lstrip('_'), target, settings=_GORILLA_SETTINGS))


def main():
    mlflow.set_tracking_uri(_TRACKING_URL)
    apply_patches()


main()
