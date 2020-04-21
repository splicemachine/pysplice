from os import environ as env_vars, popen as rbash, system as bash, remove
from sys import getsizeof
from shutil import rmtree
from pickle import dumps as save_pickle_string, loads as load_pickle_string
from io import BytesIO
from h5py import File as h5_file
import re

from pyspark.ml.base import Model as SparkModel
from tensorflow.keras.models import load_model as load_kr_model
from py4j.java_gateway import java_import

from splicemachine.spark.constants import SQL_TYPES
from splicemachine.mlflow_support.constants import *
from mleap.pyspark.spark_support import SimpleSparkSerializer

import h2o

from pyspark.ml.pipeline import PipelineModel

class SpliceMachineException(Exception):
    pass


class SQL:
    MLMANAGER_SCHEMA = 'MLMANAGER'
    ARTIFACT_INSERT_SQL = f'INSERT INTO {MLMANAGER_SCHEMA}.ARTIFACTS (run_uuid, name, "size", "binary", file_extension) VALUES (?, ?, ?, ?, ?)'
    ARTIFACT_RETRIEVAL_SQL = 'SELECT "binary", file_extension FROM ' + f'{MLMANAGER_SCHEMA}.' + 'ARTIFACTS WHERE name=\'{name}\' ' \
                                                                                'AND run_uuid=\'{runid}\''
    MODEL_INSERT_SQL = f'INSERT INTO {MLMANAGER_SCHEMA}.MODELS(RUN_UUID, MODEL, LIBRARY, "version") VALUES (?, ?, ?, ?)'
    MODEL_RETRIEVAL_SQL = 'SELECT MODEL FROM {MLMANAGER_SCHEMA}.MODELS WHERE RUN_UUID=\'{run_uuid}\''


class H2OUtils:
    @staticmethod
    def prep_model_for_deployment(splice_context, model, classes, run_id):
        """
        All preprocessing steps to prepare for in DB deployment. Get the mleap model, get class labels
        :param fittedPipe:
        :param df:
        :param classes:
        :return:  model (mleap), modelType (Enum), classes (List(str))
        """

        # Get the Mleap model and insert it into the MODELS table
        h2omojo, rawmojo = H2OUtils.get_h2omojo_model(splice_context, model)
        H2OUtils.insert_h2omojo_model(splice_context, run_id, h2omojo)

        # Get model type
        modelType, model_category = H2OUtils.get_model_type(h2omojo)
        if classes:
            if modelType not in (H2OModelType.KEY_VALUE_RETURN, H2OModelType.CLASSIFICATION):
                print('Prediction labels found but model is not type Classification. Removing labels')
                classes = None
            else:
                # handling spaces in class names
                classes = [c.replace(' ', '_') for c in classes]
                print(
                    f'Prediction labels found. Using {classes} as labels for predictions {list(range(0, len(classes)))} respectively')
        else:
            if modelType == H2OModelType.CLASSIFICATION:
                # Add a column for each class of the prediction to output the probability of the prediction
                classes = [f'p{i}' for i in list(rawmojo.getDomainValues(rawmojo.getResponseIdx()))]
            elif modelType == H2OModelType.KEY_VALUE_RETURN:
                # These types have defined outputs, and we can preformat the column names
                if model_category == 'AutoEncoder': # The input columns are the output columns (reconstruction)
                    classes = [f'{i}_reconstr' for i in list(rawmojo.getNames())]
                    if 'DeeplearningMojoModel' in rawmojo.getClass().toString(): # This class of autoencoder returns an MSE as well
                        classes.append('MSE')
                elif model_category == 'TargetEncoder':
                    classes = list(rawmojo.getNames())
                    classes.remove(rawmojo.getResponseName()) # This is the label we are training on
                    classes = [f'{i}_te' for i in classes]
                elif model_category == 'DimReduction':
                    classes = [f'PC{i}' for i in range(model.k)]
                elif model_category == 'WordEmbedding': # We create a nXm columns
                                                        # n = vector dimension, m = number of word inputs
                    classes = [f'{j}_C{i}' for i in range(rawmojo.getVecSize()) for j in rawmojo.getNames()]
                elif model_category == 'AnomalyDetection':
                    classes = ['score', 'normalizedScore']

        return modelType, classes

    @staticmethod
    def get_model_type(h2omojo):
        cat = h2omojo.getModelCategory().toString()
        if cat == 'Regression':
            modelType = H2OModelType.REGRESSION
        elif cat in ('HGLMRegression', 'Clustering'):
            modelType = H2OModelType.SINGULAR
        elif cat in ('Binomial', 'Multinomial', 'Ordinal'):
            modelType = H2OModelType.CLASSIFICATION
        elif cat in ('AutoEncoder', 'TargetEncoder', 'DimReduction', 'WordEmbedding', 'AnomalyDetection'):
            modelType = H2OModelType.KEY_VALUE_RETURN
        else:
            raise SpliceMachineException("H2O model is not supported! Only models with MOJOs are currently supported.")
        return modelType, cat


    @staticmethod
    def get_h2omojo_model(splice_context, model):
        jvm = splice_context.jvm
        java_import(jvm, "java.io.{BinaryOutputStream, ObjectOutputStream, ByteArrayInputStream}")
        java_import(jvm, 'hex.genmodel.easy.EasyPredictModelWrapper')
        java_import(jvm, 'hex.genmodel.MojoModel')
        model_path = model.download_mojo('/tmp/model.zip')
        raw_mojo = jvm.MojoModel.load(model_path)
        java_mojo_c = jvm.EasyPredictModelWrapper.Config().setModel(raw_mojo)
        java_mojo = jvm.EasyPredictModelWrapper(java_mojo_c)
        remove('/tmp/model.zip')
        return java_mojo, raw_mojo

    @staticmethod
    def log_h2o_model(splice_context, model, name, run_id):
        model_path = h2o.save_model(model=model, path='/tmp/model', force=True)
        with open(model_path, 'rb') as artifact:
            byte_stream = bytearray(bytes(artifact.read()))
        insert_artifact(splice_context, name, byte_stream, run_id, file_ext=FileExtensions.h2o)
        rmtree('/tmp/model')

    @staticmethod
    def load_h2o_model(model_blob):
        with open('/tmp/model', 'wb') as file:
            file.write(model_blob)
        model = h2o.load_model('/tmp/model')
        remove('/tmp/model')
        return model

    @staticmethod
    def insert_h2omojo_model(splice_context, run_id, model):
        model_exists = splice_context.df(
        f'select count(*) from {SQL.MLMANAGER_SCHEMA}.models where RUN_UUID=\'{run_id}\'').collect()[0][0]
        if model_exists:
            print(
            'A model with this ID already exists in the table. We are NOT replacing it. We will use the currently existing model.\nTo replace, use a new run_id')
        else:
            baos = splice_context.jvm.java.io.ByteArrayOutputStream()
            oos = splice_context.jvm.java.io.ObjectOutputStream(baos)
            oos.writeObject(model)
            oos.flush()
            oos.close()
            byte_array = baos.toByteArray()
            insert_model(splice_context, run_id, byte_array, 'h2omojo', h2o.__version__)


class SKUtils:
    @staticmethod
    def log_sklearn_model(splice_context, model, name, run_id):
        byte_stream = save_pickle_string(model)
        insert_artifact(splice_context, name, byte_stream, run_id, file_ext=FileExtensions.sklearn)

    @staticmethod
    def load_sklearn_model(model_blob):
        return load_pickle_string(model_blob)

class KerasUtils:
    @staticmethod
    def log_keras_model(splice_context, model, name, run_id):
        model.save('/tmp/model.h5')
        with open('/tmp/model.h5', 'rb') as f:
            byte_stream = bytearray(bytes(f.read()))
        insert_artifact(splice_context, name, byte_stream, run_id, file_ext=FileExtensions.keras)
        remove('/tmp/model.h5')

    @staticmethod
    def load_keras_model(model_blob):
        hfile = h5_file(BytesIO(model_blob), 'r')
        return load_kr_model(hfile)

class SparkUtils:
    @staticmethod
    def get_stages(pipeline):
        """
        Extract the stages from a fit or unfit pipeline
        :param pipeline: a fit or unfit Spark pipeline
        :return: stages list
        """
        if hasattr(pipeline, 'getStages'):
            return pipeline.getStages()  # fit pipeline
        return pipeline.stages  # unfit pipeline

    @staticmethod
    def get_cols(transformer, get_input=True):
        """
        Get columns from a transformer
        :param transformer: the transformer (fit or unfit)
        :param get_input: whether or not to return the input columns
        :return: a list of either input or output columns
        """
        col_type = 'Input' if get_input else 'Output'
        if hasattr(transformer, 'get' + col_type + 'Col'):  # single transformer 1:1 (regular)
            col_function = transformer.getInputCol if get_input else transformer.getOutputCol
            return [
                col_function()] if get_input else col_function()  # so we don't need to change code for
            # a single transformer
        elif hasattr(transformer, col_type + "Col"):  # single transformer 1:1 (vec)
            return getattr(transformer, col_type + "Col")
        elif get_input and hasattr(transformer,
                                   'get' + col_type + 'Cols'):  # multi ple transformer n:1 (regular)
            return transformer.getInputCols()  # we can never have > 1 output column (goes to vector)
        elif get_input and hasattr(transformer, col_type + 'Cols'):  # multiple transformer n:1 (vec)
            return getattr(transformer, col_type + "Cols")
        else:
            print(
                "Warning: Transformer " + str(
                    transformer) + " could not be parsed. If this is a model, this is expected.")

    @staticmethod
    def readable_pipeline_stage(pipeline_stage):
        """
        Get a readable version of the Pipeline stage
        (without the memory address)
        :param pipeline_stage: the name of the stage to parse
        """
        if '_' in str(pipeline_stage):
            return str(pipeline_stage).split('_')[0]
        return str(pipeline_stage)

    @staticmethod
    def is_spark_pipeline(spark_object):
        """
        Returns whether or not the given
        object is a spark pipeline. If it
        is a model, it will return True, if it is a
        pipeline model is will return False.
        Otherwise, it will throw an exception
        :param spark_object: (Model) Spark object to check
        :return: (bool) whether or not the object is a model
        :exception: (Exception) throws an error if it is not either
        """
        if isinstance(spark_object, PipelineModel):
            return True

        if isinstance(spark_object, SparkModel):
            return False

        raise Exception("The model supplied does not appear to be a Spark Model!")

    @staticmethod
    def get_model_stage(pipeline):
        """"
        Gets the Model stage of a FIT PipelineModel
        """
        for i in SparkUtils.get_stages(pipeline):
            # FIXME: We need a better way to determine if a stage is a model
            if 'Model' in str(i.__class__) and i.__module__.split('.')[-1] in ['clustering', 'classification',
                                                                               'regression']:
                return i
        raise AttributeError('Could not find model stage in Pipeline! Is this a fitted spark Pipeline?')

    @staticmethod
    def parse_string_parameters(string_parameters):
        """
        Parse string rendering of extractParamMap
        :param string_parameters:
        :return:
        """
        parameters = {}
        parsed_mapping = str(string_parameters).replace("{", "").replace(
            "}", "").replace("\n", "").replace("\t", "").split(',')

        for mapping in parsed_mapping:
            param = mapping[mapping.index('-') + 1:].split(':')[0]
            value = mapping[mapping.index('-') + 1:].split(':')[1]

            parameters[param.strip()] = value.strip()
        return parameters

    @staticmethod
    def retrieve_artifact_stream(splice_context, run_id, name):
        """
        Retrieve the binary stream for a given
        artifact with the specified name and run id
        :param splice_context: splice context 
        :param run_id: (str) the run id for the run
            that the artifact belongs to
        :param name: (str) the name of the artifact
        :return: (bytearray(byte)) byte array from BLOB
        """
        try:
            return splice_context.df(
                SQL.ARTIFACT_RETRIEVAL_SQL.format(name=name, runid=run_id)
            ).collect()[0]
        except IndexError as e:
            raise Exception(f"Unable to find the artifact with the given run id {run_id} and name {name}")

    @staticmethod
    def get_feature_vector_columns(fittedPipe):
        """
        Gets the input columns from the VectorAssembler stage of a fitted Pipeline
        """
        vec_stage = None
        for i in fittedPipe.stages:
            if 'VectorAssembler' in str(i):
                vec_stage = i
                break
        return SparkUtils.get_cols(vec_stage)

    @staticmethod
    def get_num_classes(pipeline_or_model):
        '''
        Tries to find the number of classes in a Pipeline or Model object
        :param pipeline_or_model: The Pipeline or Model object
        '''
        if 'pipeline' in str(pipeline_or_model.__class__).lower():
            model = SparkUtils.get_model_stage(pipeline_or_model)
        else:
            model = pipeline_or_model
        num_classes = model.numClasses if model.hasParam('numClasses') else model.numClasses if hasattr(model,
                                                                                                        'numClasses') else model.summary.k
        return num_classes

    @staticmethod
    def get_model_type(pipeline_or_model):
        """
        Takes a fitted Spark Pipeline or Model and determines if it is a Regression, Classification, or Clustering model
        """
        model = SparkUtils.get_model_stage(pipeline_or_model)

        if model.__module__ == 'pyspark.ml.classification':
            m_type = SparkModelType.CLASSIFICATION
        elif model.__module__ == 'pyspark.ml.regression':
            m_type = SparkModelType.REGRESSION
        elif model.__module__ == 'pyspark.ml.clustering':
            if 'probabilityCol' in model.explainParams():
                m_type = SparkModelType.CLUSTERING_WITH_PROB
            else:
                m_type = SparkModelType.CLUSTERING_WO_PROB

        return m_type
    @staticmethod
    def log_spark_model(splice_context, model, name, run_id):
        jvm = splice_context.jvm
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
        insert_artifact(splice_context, name, baos.toByteArray(), run_id,
                    file_ext='spark')  # write the byte stream to the db as a BLOB

    @staticmethod
    def load_spark_model(splice_ctx, spark_pipeline_blob):
        jvm = splice_ctx.jvm
        bis = jvm.java.io.ByteArrayInputStream(spark_pipeline_blob)
        ois = jvm.java.io.ObjectInputStream(bis)
        pipeline = PipelineModel._from_java(ois.readObject())  # convert object from Java
        # PipelineModel to Python PipelineModel
        ois.close()

        if len(pipeline.stages) == 1 and not SparkUtils.is_spark_pipeline(pipeline.stages[0]):
            pipeline = pipeline.stages[0]

        return pipeline

    @staticmethod
    def prep_model_for_deployment(splice_context, fittedPipe, df, classes, run_id):
        """
        All preprocessing steps to prepare for in DB deployment. Get the mleap model, get class labels
        :param fittedPipe:
        :param df:
        :param classes:
        :return:  model (mleap), modelType (Enum), classes (List(str))
        """
        # Get model type
        modelType = SparkUtils.get_model_type(fittedPipe)
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
        mleap_model = get_mleap_model(splice_context, fittedPipe, df, run_id)
        insert_mleap_model(splice_context, run_id, mleap_model)

        return modelType, classes

def find_inputs_by_output(dictionary, value):
    """
    Find the input columns for a given output column
    :param dictionary: dictionary to search
    :param value: column
    :return: None if not found, otherwise first column
    """
    keys = []
    for key in dictionary:
        if dictionary[key][1] == value:  # output column is always the last one
            keys.append(key)
    return keys


def get_user():
    """
    Get the current logged in user to
    Jupyter
    :return: (str) name of the logged in user
    """
    try:
        uname = env_vars.get('JUPYTERHUB_USER') or env_vars['USER']
        return uname
    except KeyError:
        raise Exception(
            "Could not determine current running user. Running MLManager outside of Splice Machine"
            " Cloud Jupyter is currently unsupported")


def insert_model(splice_context, run_id, byte_array, library, version):
    """
    Insert a serialized model into the Mlmanager models table
    :param splice_context: pysplicectx
    :param run_id: mlflow run id
    :param byte_array: byte array
    :param library: The library of the model (mleap, h2omojo etc)
    :param version: The version of the library
    """
    db_connection = splice_context.getConnection()
    file_size = getsizeof(byte_array)
    print(f"Saving model of size: {file_size / 1000.0} KB to Splice Machine DB")
    binary_input_stream = splice_context.jvm.java.io.ByteArrayInputStream(byte_array)

    prepared_statement = db_connection.prepareStatement(SQL.MODEL_INSERT_SQL)
    prepared_statement.setString(1, run_id)
    prepared_statement.setBinaryStream(2, binary_input_stream)
    prepared_statement.setString(3, library)
    prepared_statement.setString(4, version)

    prepared_statement.execute()
    prepared_statement.close()


def insert_artifact(splice_context, name, byte_array, run_uuid, file_ext=None):
    """
    :param name: (str) the path to store the binary
        under (with respect to the current run)
    :param byte_array: (byte[]) Java byte array
    :param run_uuid: run uuid for the run
    :param file_ext: (str) the file extension of the model (used for downloading)
    """
    db_connection = splice_context.getConnection()
    file_size = getsizeof(byte_array)
    print(f"Saving artifact of size: {file_size / 1000.0} KB to Splice Machine DB")
    binary_input_stream = splice_context.jvm.java.io.ByteArrayInputStream(byte_array)
    prepared_statement = db_connection.prepareStatement(SQL.ARTIFACT_INSERT_SQL)
    prepared_statement.setString(1, run_uuid)
    prepared_statement.setString(2, name)
    prepared_statement.setInt(3, file_size)
    prepared_statement.setBinaryStream(4, binary_input_stream)
    prepared_statement.setString(5, file_ext)

    prepared_statement.execute()
    prepared_statement.close()


def get_pod_uri(pod, port, _testing=False):
    """
    Get address of MLFlow Container for Kubernetes
    """

    if _testing:
        url = f"http://{pod}:{port}"  # mlflow docker container endpoint

    elif 'MLFLOW_URL' in env_vars:
        url = env_vars['MLFLOW_URL'].rstrip(':5001')
        url += f':{port}'  # 5001 or 5003 for tracking or deployment
    else:
        raise KeyError(
            "Uh Oh! MLFLOW_URL variable was not found... are you running in the Cloud service?")
    return url


def get_mleap_model(splice_context, fittedPipe, df, run_id: str):
    """
    Turns a fitted Spark Pipeline into an Mleap Transformer
    :param splice_context: pysplicectx
    :param fittedPipe: Fitted Spark Pipeline
    :param df: A TRANSFORMED dataframe. ie a dataframe that the pipeline has called .transform() on
    :param run_id: (str) the MLFlow run associated with the model
    """
    SimpleSparkSerializer()  # Adds the serializeToBundle function from Mleap
    if 'tmp' not in rbash('ls /').read():
        bash('mkdir /tmp')
    # Serialize the Spark model into Mleap format
    if f'{run_id}.zip' in rbash('ls /tmp').read():
        remove(f'/tmp/{run_id}.zip')
    fittedPipe.serializeToBundle(f"jar:file:///tmp/{run_id}.zip", df)

    jvm = splice_context.jvm
    java_import(jvm, "com.splicemachine.mlrunner.FileRetriever")
    obj = jvm.FileRetriever.loadBundle(f'jar:file:///tmp/{run_id}.zip')
    remove(f'/tmp/{run_id}.zip"')
    return obj


def insert_mleap_model(splice_context, run_id, model):
    """
    Insert an MLeap Transformer model into the database as a Blob
    :param splice_context: pysplicectx
    :param model_id: (str) the mlflow run_id that the model is associated with
        (with respect to the current run)
    :param model: (Transformer) the fitted Mleap Transformer (pipeline)
    """

    # If a model with this run_id already exists in the table, gracefully fail
    # May be faster to use prepared statement
    model_exists = splice_context.df(
        f'select count(*) from {SQL.MLMANAGER_SCHEMA}.models where RUN_UUID=\'{run_id}\'').collect()[0][0]
    if model_exists:
        print(
            'A model with this ID already exists in the table. We are NOT replacing it. We will use the currently existing model.\nTo replace, use a new run_id')

    else:
        # Serialize Mleap model to BLOB
        baos = splice_context.jvm.java.io.ByteArrayOutputStream()
        oos = splice_context.jvm.java.io.ObjectOutputStream(baos)
        oos.writeObject(model)
        oos.flush()
        oos.close()
        byte_array = baos.toByteArray()
        insert_model(splice_context, run_id, byte_array, 'mleap', MLEAP_VERSION)


def validate_primary_key(primary_key):
    """
    Function to validate the primary key passed by the user conforms to SQL
    :param primary_key: List[Tuple[str,str]] column name, SQL datatype for the primary key(s) of the table
    """
    regex = re.compile('[^a-zA-Z]')
    for i in primary_key:
        sql_datatype = regex.sub('', i[1]).upper()
        if sql_datatype not in SQL_TYPES:
            raise ValueError(f'Primary key parameter {i} does not conform to SQL type.'
                             f'Value {primary_key[i][1]} should be a SQL type but isn\'t')


def create_data_table(splice_context, schema_table_name, schema_str, primary_key,
                        verbose):
    """
    Creates the table that holds the columns of the feature vector as well as a unique MOMENT_ID
    :param splice_context: pysplicectx
    :param schema_table_name: (str) the schema.table to create the table under
    :param schema_str: (str) the structure of the schema of the table as a string (col_name TYPE,)
    :param primary_key: List[Tuple[str,str]] column name, SQL datatype for the primary key(s) of the table
    :param verbose: (bool) whether to print the SQL query
    """
    if splice_context.tableExists(schema_table_name):
        raise SpliceMachineException(
            f'A model has already been deployed to table {schema_table_name}. We currently only support deploying 1 model per table')
    SQL_TABLE = f'CREATE TABLE {schema_table_name} (\n' + schema_str

    pk_cols = ''
    for i in primary_key:
        # If pk is already in the schema_string, don't add another column. PK may be an existing value
        if i[0] not in schema_str:
            SQL_TABLE += f'\t{i[0]} {i[1]},\n'
        pk_cols += f'{i[0]},'
    SQL_TABLE += f'\tPRIMARY KEY({pk_cols.rstrip(",")})\n)'

    if verbose: print('\n', SQL_TABLE, end='\n\n')
    splice_context.execute(SQL_TABLE)


def create_data_preds_table(splice_context, run_id, schema_table_name, classes, primary_key,
                              modelType, verbose):
    """
    Creates the data prediction table that holds the prediction for the rows of the data table
    :param splice_context: pysplicectx
    :param schema_table_name: (str) the schema.table to create the table under
    :param run_id: (str) the run_id for this model
    :param classes: (List[str]) the labels of the model (if they exist)
    :param primary_key: List[Tuple[str,str]] column name, SQL datatype for the primary key(s) of the table
    :param modelType: (ModelType) Whether the model is a Regression, Classification or Clustering (with/without probabilities)
    :param verbose: (bool) whether to print the SQL query
    Regression models output a DOUBLE as the prediction field with no probabilities
    Classification models and Certain Clustering models have probabilities associated with them, so we need to handle the extra columns holding those probabilities
    Clustering models without probabilities return only an INT.
    The database is strongly typed so we need to specify the output type for each ModelType
    """
    SQL_PRED_TABLE = f'''CREATE TABLE {schema_table_name}_PREDS (
        \tCUR_USER VARCHAR(50) DEFAULT CURRENT_USER,
        \tEVAL_TIME TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        \tRUN_ID VARCHAR(50) DEFAULT \'{run_id}\',
        '''
    pk_cols = ''
    for i in primary_key:
        SQL_PRED_TABLE += f'\t{i[0]} {i[1]},\n'
        pk_cols += f'{i[0]},'

    if modelType in (SparkModelType.REGRESSION, H2OModelType.REGRESSION):
        SQL_PRED_TABLE += '\tPREDICTION DOUBLE,\n'

    elif modelType in (SparkModelType.CLASSIFICATION, SparkModelType.CLUSTERING_WITH_PROB,
                       H2OModelType.CLASSIFICATION):
        SQL_PRED_TABLE += '\tPREDICTION VARCHAR(250),\n'
        for i in classes:
            SQL_PRED_TABLE += f'\t{i} DOUBLE,\n'

    elif modelType == H2OModelType.KEY_VALUE_RETURN:
        for i in classes:
            SQL_PRED_TABLE += f'\t{i} DOUBLE,\n'

    elif modelType in (SparkModelType.CLUSTERING_WO_PROB, H2OModelType.SINGULAR):
        SQL_PRED_TABLE += '\tPREDICTION INT,\n'

    SQL_PRED_TABLE += f'\tPRIMARY KEY({pk_cols.rstrip(",")})\n)'

    if verbose:
        print()
        print(SQL_PRED_TABLE, end='\n\n')
    splice_context.execute(SQL_PRED_TABLE)

def create_vti_prediction_trigger(splice_context, schema_table_name, run_id, feature_columns, schema_types, schema_str, primary_key, classes, verbose):


    prediction_call = "new com.splicemachine.mlrunner.MLRunner('key_value', '{run_id}', {raw_data}, '{schema_str}')"
    SQL_PRED_TRIGGER = f'CREATE TRIGGER runModel_{schema_table_name.replace(".", "_")}_{run_id}\n \tAFTER INSERT\n ' \
                       f'\tON {schema_table_name}\n \tREFERENCING NEW AS NEWROW\n \tFOR EACH ROW\n \t\tINSERT INTO ' \
                       f'{schema_table_name}_PREDS('
    pk_vals = ''
    for i in primary_key:
        SQL_PRED_TRIGGER += f'{i[0]},'
        pk_vals += f'\tNEWROW.{i[0]},'

    output_column_names = '' # Names of the output columns from the model
    output_cols_VTI_reference = '' # Names references from the VTI (ie b.COL_NAME)
    output_cols_schema = ''  # Names with their datatypes (always DOUBLE)
    for i in classes:
        output_column_names += f'{i},'
        output_cols_VTI_reference += f'b.{i},'
        output_cols_schema += f'{i} DOUBLE,'

    raw_data = ''
    for i, col in enumerate(feature_columns):
        raw_data += '||' if i != 0 else ''
        inner_cast = f'CAST(NEWROW.{col} as DECIMAL(38,10))' if schema_types[str(col)] in {'FloatType', 'DoubleType',
                                                                                           'DecimalType'} else f'NEWROW.{col}'
        raw_data += f'TRIM(CAST({inner_cast} as CHAR(41)))||\',\''

    # Cleanup + schema for PREDICT call
    raw_data = raw_data[:-5].lstrip('||')
    schema_str_pred_call = schema_str.replace('\t', '').replace('\n','').rstrip(',')
    prediction_call = prediction_call.format(run_id=run_id, raw_data=raw_data, schema_str=schema_str_pred_call)

    SQL_PRED_TRIGGER += f'{output_column_names[:-1]}) SELECT {pk_vals} {output_cols_VTI_reference[:-1]} FROM {prediction_call}' \
                        f' as b ({output_cols_schema[:-1]})'



    if verbose:
        print()
        print(SQL_PRED_TRIGGER, end='\n\n')
    splice_context.execute(SQL_PRED_TRIGGER.replace('\n', ' ').replace('\t', ' '))

def create_prediction_trigger(splice_context, schema_table_name, run_id, feature_columns,
                                schema_types, schema_str, primary_key,
                                modelType, verbose):
    """
    Creates the trigger that calls the model on data row insert. This trigger will call predict when a new row is inserted into the data table
    and insert the result into the predictions table.
    :param splice_context: pysplicectx
    :param schema_table_name: (str) the schema.table to create the table under
    :param run_id: (str) the run_id to deploy the model under
    :param feature_columns: (List[str]) the original features that are transformed into the final feature vector
    :param schema_types: (Dict[str, str]) a mapping of feature column to data type
    :param schema_str: (str) the structure of the schema of the table as a string (col_name TYPE,)
    :param primary_key: List[Tuple[str,str]] column name, SQL datatype for the primary key(s) of the table
    :param modelType: (ModelType) Whether the model is a Regression, Classification or Clustering (with/without probabilities)
    :param verbose: (bool) whether to print the SQL query
    """

    if modelType in (SparkModelType.CLASSIFICATION, H2OModelType.CLASSIFICATION):
        prediction_call = 'MLMANAGER.PREDICT_CLASSIFICATION'
    elif modelType in (SparkModelType.REGRESSION, H2OModelType.REGRESSION):
        prediction_call = 'MLMANAGER.PREDICT_REGRESSION'
    elif modelType == SparkModelType.CLUSTERING_WITH_PROB:
        prediction_call = 'MLMANAGER.PREDICT_CLUSTER_PROBABILITIES'
    elif modelType in (SparkModelType.CLUSTERING_WO_PROB, H2OModelType.SINGULAR):
        prediction_call = 'MLMANAGER.PREDICT_CLUSTER'
    elif modelType == H2OModelType.KEY_VALUE_RETURN:
        prediction_call = 'MLMANAGER.PREDICT_KEY_VALUE'


    SQL_PRED_TRIGGER = f'CREATE TRIGGER runModel_{schema_table_name.replace(".", "_")}_{run_id}\n \tAFTER INSERT\n ' \
                       f'\tON {schema_table_name}\n \tREFERENCING NEW AS NEWROW\n \tFOR EACH ROW\n \t\tINSERT INTO ' \
                       f'{schema_table_name}_PREDS('
    pk_vals = ''
    for i in primary_key:
        SQL_PRED_TRIGGER += f'\t{i[0]},'
        pk_vals += f'\tNEWROW.{i[0]},'

    SQL_PRED_TRIGGER += f'PREDICTION) VALUES({pk_vals}' + f'{prediction_call}(\'{run_id}\','

    for i, col in enumerate(feature_columns):
        SQL_PRED_TRIGGER += '||' if i != 0 else ''
        inner_cast = f'CAST(NEWROW.{col} as DECIMAL(38,10))' if schema_types[str(col)] in {'FloatType', 'DoubleType',
                                                                                           'DecimalType'} else f'NEWROW.{col}'
        SQL_PRED_TRIGGER += f'TRIM(CAST({inner_cast} as CHAR(41)))||\',\''

    # Cleanup + schema for PREDICT call
    SQL_PRED_TRIGGER = SQL_PRED_TRIGGER[:-5].lstrip('||') + ',\n\'' + schema_str.replace('\t', '').replace('\n',
                                                                                                    '').rstrip(',') + '\'))'
    if verbose:
        print()
        print(SQL_PRED_TRIGGER, end='\n\n')
    splice_context.execute(SQL_PRED_TRIGGER.replace('\n', ' ').replace('\t', ' '))


def create_parsing_trigger(splice_context, schema_table_name, primary_key, run_id,
                             classes, modelType, verbose):
    """
    Creates the secondary trigger that parses the results of the first trigger and updates the prediction row populating the relevant columns
    :param splice_context: splice context specified in mlflow
    :param schema_table_name: (str) the schema.table to create the table under
    :param primary_key: List[Tuple[str,str]] column name, SQL datatype for the primary key(s) of the table
    :param run_id: (str) the run_id to deploy the model under
    :param classes: (List[str]) the labels of the model (if they exist)
    :param modelType: (Enum) the model type (H2OModelType or SparkModelType)
    :param verbose: (bool) whether to print the SQL query
    """
    SQL_PARSE_TRIGGER = f'CREATE TRIGGER PARSERESULT_{schema_table_name.replace(".", "_")}_{run_id}' \
                        f'\n \tAFTER INSERT\n \tON {schema_table_name}_PREDS\n \tREFERENCING NEW AS NEWROW\n' \
                        f' \tFOR EACH ROW\n \t\tUPDATE {schema_table_name}_PREDS set '
    set_prediction_case_str = 'PREDICTION=\n\t\tCASE\n'
    for i, c in enumerate(classes):
        SQL_PARSE_TRIGGER += f'{c}=MLMANAGER.PARSEPROBS(NEWROW.prediction,{i}),'
        set_prediction_case_str += f'\t\tWHEN MLMANAGER.GETPREDICTION(NEWROW.prediction)={i} then \'{c}\'\n'
    set_prediction_case_str += '\t\tEND'
    if modelType == H2OModelType.KEY_VALUE_RETURN: # These models don't have an actual prediction
        SQL_PARSE_TRIGGER = SQL_PARSE_TRIGGER[:-1] + ' WHERE'
    else:
        SQL_PARSE_TRIGGER += set_prediction_case_str + ' WHERE'

    for i in primary_key:
        SQL_PARSE_TRIGGER += f' {i[0]}=NEWROW.{i[0]} AND'
    SQL_PARSE_TRIGGER = SQL_PARSE_TRIGGER.replace(' AND', '')

    if verbose:
        print()
        print(SQL_PARSE_TRIGGER, end='\n\n')
    splice_context.execute(SQL_PARSE_TRIGGER.replace('\n', ' ').replace('\t', ' '))


def drop_tables_on_failure(splice_context, schema_table_name, run_id) -> None:
    """
    Drop the tables if the db deployment fails
    """
    splice_context.execute(f'DROP TABLE IF EXISTS {schema_table_name}')
    splice_context.execute(f'DROP TABLE IF EXISTS {schema_table_name}_preds')
    splice_context.execute(f'DELETE FROM {SQL.MLMANAGER_SCHEMA}.MODELS WHERE RUN_UUID=\'{run_id}\'')
