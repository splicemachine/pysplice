from os import environ as env_vars, popen as rbash, system as bash, remove
from sys import getsizeof
from shutil import rmtree
from cloudpickle import dumps as save_pickle_string, loads as load_pickle_string
from io import BytesIO
from h5py import File as h5_file
from py4j.java_gateway import java_import
from inspect import signature as get_model_params
import re

from ..spark.context import PySpliceContext

from pyspark.ml.base import Model as SparkModel
from pyspark.ml.feature import IndexToString
from pyspark.ml.wrapper import JavaModel
from pyspark.ml import classification as spark_classification, regression as spark_regression, \
    clustering as spark_clustering, recommendation as spark_recommendation
from pyspark.sql.types import StructType
from pyspark.sql.dataframe import DataFrame as SparkDF
from pandas.core.frame import DataFrame as PandasDF

from splicemachine.spark.constants import SQL_TYPES
from splicemachine.mlflow_support.constants import *
from mleap.pyspark.spark_support import SimpleSparkSerializer
from mleap.version import __version__ as MLEAP_VERSION

import sklearn.base
from sklearn import __version__ as sklearn_version
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.base import BaseEstimator as ScikitModel
from tensorflow.keras.models import load_model as load_kr_model
from tensorflow.keras import Model as KerasModel
from tensorflow.keras import __version__ as KERAS_VERSION

import h2o
from h2o.estimators.estimator_base import ModelBase as H2OModel
from pyspark.ml.pipeline import PipelineModel
from typing import List, Dict, Tuple

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
    def prep_model_for_deployment(splice_context: PySpliceContext,
                                  model: H2OModel,
                                  classes: List[str],
                                  run_id: str,
                                  df: SparkDF or None,
                                  pred_threshold: float or None,
                                  sklearn_args: Dict[str,str] or None) -> (H2OModelType, List[str]):
        """
        Gets the H2O mojo model
        Gets the model type
        Inserts the raw model into the MODELS table
        gets the classes (if applicable)
        returns the model type and classes
        :param splice_context:
        :param model:
        :param classes:
        :param run_id:
        :return:
        """

        # Get the H2O MOJO model and insert it into the MODELS table
        h2omojo, rawmojo = H2OUtils.get_h2omojo_model(splice_context, model)
        model_already_exists = H2OUtils.insert_h2omojo_model(splice_context, run_id, h2omojo)

        # Get model type
        model_type, model_category = H2OUtils.get_model_type(h2omojo)
        if classes:
            if model_type not in (H2OModelType.KEY_VALUE, H2OModelType.CLASSIFICATION):
                print('Prediction labels found but model type does not support classes. Removing labels')
                classes = None
            else:
                # handling spaces in class names
                classes = [c.replace(' ', '_') for c in classes]
                print(
                    f'Prediction labels found. '
                    f'Using {classes} as labels for predictions {list(range(0, len(classes)))} respectively'
                )
        else:
            if model_type == H2OModelType.CLASSIFICATION:
                # Add a column for each class of the prediction to output the probability of the prediction
                classes = [f'p{i}' for i in list(rawmojo.getDomainValues(rawmojo.getResponseIdx()))]
            elif model_type == H2OModelType.KEY_VALUE:
                # These types have defined outputs, and we can preformat the column names
                if model_category == 'AutoEncoder':  # The input columns are the output columns (reconstruction)
                    classes = [f'{i}_reconstr' for i in list(rawmojo.getNames())]
                    if 'DeeplearningMojoModel' in rawmojo.getClass().toString():  # This class of autoencoder returns an MSE as well
                        classes.append('MSE')
                elif model_category == 'TargetEncoder':
                    classes = list(rawmojo.getNames())
                    classes.remove(rawmojo.getResponseName())  # This is the label we are training on
                    classes = [f'{i}_te' for i in classes]
                elif model_category == 'DimReduction':
                    classes = [f'PC{i}' for i in range(model.k)]
                elif model_category == 'WordEmbedding':  # We create a nXm columns
                    # n = vector dimension, m = number of word inputs
                    classes = [f'{j}_C{i}' for i in range(rawmojo.getVecSize()) for j in rawmojo.getNames()]
                elif model_category == 'AnomalyDetection':
                    classes = ['score', 'normalizedScore']

        return model_type, classes, model_already_exists

    @staticmethod
    def get_model_type(h2omojo: object) -> (H2OModelType, str):
        cat = h2omojo.getModelCategory().toString()
        if cat == 'Regression':
            model_type = H2OModelType.REGRESSION
        elif cat in ('HGLMRegression', 'Clustering'):
            model_type = H2OModelType.SINGULAR
        elif cat in ('Binomial', 'Multinomial', 'Ordinal'):
            model_type = H2OModelType.CLASSIFICATION
        elif cat in ('AutoEncoder', 'TargetEncoder', 'DimReduction', 'WordEmbedding', 'AnomalyDetection'):
            model_type = H2OModelType.KEY_VALUE
        else:
            raise SpliceMachineException(f"H2O model {cat} is not supported! Only models with MOJOs are currently supported.")
        return model_type, cat

    @staticmethod
    def get_h2omojo_model(splice_context: PySpliceContext, model: H2OModel) -> (object, object):
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
    def log_h2o_model(splice_context: PySpliceContext, model: object, name: str, run_id: str) -> None:
        model_path = h2o.save_model(model=model, path='/tmp/model', force=True)
        with open(model_path, 'rb') as artifact:
            byte_stream = bytearray(bytes(artifact.read()))
        insert_artifact(splice_context, name, byte_stream, run_id, file_ext=FileExtensions.h2o)
        rmtree('/tmp/model')

    @staticmethod
    def load_h2o_model(model_blob: bytes) -> H2OModel:
        with open('/tmp/model', 'wb') as file:
            file.write(model_blob)
        model = h2o.load_model('/tmp/model')
        remove('/tmp/model')
        return model

    @staticmethod
    def insert_h2omojo_model(splice_context: PySpliceContext, run_id: str, model: object) -> bool:
        baos = splice_context.jvm.java.io.ByteArrayOutputStream()
        oos = splice_context.jvm.java.io.ObjectOutputStream(baos)
        oos.writeObject(model)
        oos.flush()
        oos.close()
        byte_array = baos.toByteArray()
        return insert_model(splice_context, run_id, byte_array, 'h2omojo', h2o.__version__)


class SKUtils:
    @staticmethod
    def log_sklearn_model(splice_context: PySpliceContext, model: ScikitModel, name: str, run_id: str):
        byte_stream = save_pickle_string(model)
        insert_artifact(splice_context, name, byte_stream, run_id, file_ext=FileExtensions.sklearn)

    @staticmethod
    def load_sklearn_model(model_blob: bytes):
        return load_pickle_string(model_blob)

    @staticmethod
    def insert_sklearn_model(splice_context: PySpliceContext, run_id: str, model: ScikitModel) -> bool:
        byte_stream = save_pickle_string(model)
        return insert_model(splice_context, run_id, byte_stream, 'sklearn', sklearn_version)

    @staticmethod
    def validate_sklearn_args(model: ScikitModel, sklearn_args: Dict[str, str]) -> Dict[str, str]:
        """
        Make sure sklearn args contains valid values. sklearn_args can only contain 2 keys:
        predict_call and predict_args.
        predict_call: 'predict'/'predict_proba'/'transform'
        predict_args: 'return_std'/'return_cov'

        :param model: (ScikitModel)
        :param sklearn_args: (Dict[str, str])
        :return: (Dict[str, str]) sklearn_args
        """
        exc = ''
        keys = set(sklearn_args.keys())
        if keys - {'predict_call', 'predict_args'} != set():
            exc = "You've passed in an sklearn_args key that is not valid. Valid keys are ('predict_call', 'predict_args')"
        elif len(sklearn_args) > 2:
            exc ='Only predict_call and predict_args are allowed in sklearn_args!'
        elif 'predict_call' in sklearn_args:
            p = sklearn_args['predict_call']
            if not hasattr(model, p):
                exc = f'predict_call set to {p} but function call not available in model {model}'
            if p != 'predict' and 'predict_args' in sklearn_args:
                exc = f'predict_args passed in but predict_call is {p}. This combination is not allowed'
        if 'predict_args' in sklearn_args:
            p = sklearn_args['predict_args']
            if p not in ('return_std', 'return_cov') and not isinstance(model, SKPipeline): # Pipelines have difference rules for params
                t = ('return_std', 'return_cov')
                exc = f'predict_args value is invalid. Available options are {t}'
            else:
                if isinstance(model, SKPipeline): # If we are working with a Pipeline, we want to check the last step for arguments
                    m = model.steps[-1][-1]
                    model_params = get_model_params(m.predict) if hasattr(m, 'predict') else get_model_params(m.transform)
                else:
                    model_params = get_model_params(model.predict) if hasattr(model, 'predict') else get_model_params(model.transform)
                if p not in model_params.parameters:
                    exc = f'predict_args set to {p} but that parameter is not available for this model!'
        elif sklearn_args and 'predict_args' not in sklearn_args and 'predict_call' not in sklearn_args:
                exc = f"predict_args contains invalid arguments. Valid arguments are 'predict_call' and 'predict_args'"
        if exc:
            raise SpliceMachineException(exc)
        if sklearn_args.get('predict_call') == 'predict' and 'predict_args' not in sklearn_args:
            # If the user only passed in predict, then sklearn args is effectively empty
            sklearn_args = None
        return sklearn_args

    @staticmethod
    def prep_model_for_deployment(splice_context: PySpliceContext,
                                  model: ScikitModel,
                                  classes: List[str],
                                  run_id: str,
                                  df: SparkDF or None,
                                  pred_threshold: float or None,
                                  sklearn_args: Dict[str,str] or None) -> (SklearnModelType, List[str], bool):
        """
        Preperatory steps to deploy a scikit-learn model

        :param splice_context: (PySpliceContext)
        :param model: (ScikitModel)
        :param classes: (List[str]) The label names for the model
        :param run_id: (str) the run_id being deployed under
        :param df: (SparkDF or None) the spark dataframe if necessary
        :param pred_threshold: (None) unused in this abstract method implementation
        :param sklearn_args: (Dict[str,str]) The optional sklearn args
        :return: (SklearnModelType, List[str], bool) the model_type, classes, and model_already_exists
        """

        sklearn_args = SKUtils.validate_sklearn_args(model, sklearn_args)

        model_type = SKUtils.get_model_type(model, sklearn_args)
        model_already_exists = SKUtils.insert_sklearn_model(splice_context, run_id, model)
        if classes and model_type != SklearnModelType.KEY_VALUE:
            print('Prediction labels found but model is not type Classification. Removing labels')
            classes = None

        elif classes:
            classes = [i.replace(' ', '_') for i in classes]

        elif model_type == SklearnModelType.KEY_VALUE:
            # For models that have both predict and transform functions (like LDA)
            if sklearn_args.get('predict_call') == 'transform' and hasattr(model, 'transform'):
                params = model.get_params()
                nclasses = params.get('n_clusters') or params.get('n_components') or 2
                classes = [f'C{i}' for i in range(nclasses)]
            elif 'predict_args' in sklearn_args:
                classes = ['prediction', sklearn_args.get('predict_args').lstrip('return_')]
            elif hasattr(model, 'classes_') and model.classes_.size != 0:
                classes = [f'C{i}' for i in model.classes_]
            elif hasattr(model, 'get_params') and ( hasattr(model,'n_components') or hasattr(model,'n_components') ):
                params = model.get_params()
                nclasses = params.get('n_clusters') or params.get('n_components')
                classes = [f'C{i}' for i in range(nclasses)]
            else:
                raise SpliceMachineException('Could not find class labels from model. Please pass in class labels using'
                                             'the classes parameter.')

        if sklearn_args.get('predict_call') == 'predict_proba': # We need to add a column for the actual prediction
            classes = ['prediction'] + classes
        if classes:
            print(f'Prediction labels found. Using {classes} as labels for predictions {list(range(0, len(classes)))} respectively')

        return model_type, classes, model_already_exists

    @staticmethod
    def get_pipeline_model_type(pipeline: SKPipeline) -> SklearnModelType:
        """
        Gets the type of model in the Sklearn Pipeline

        :param pipeline: The Sklearn Pipeline
        :return: SKlearnModelType
        """
        model_type = None
        for _, step in pipeline.steps[::-1]: # Go through steps backwards because model likely last step
            if isinstance(step, (sklearn.base.ClusterMixin, sklearn.base.ClassifierMixin)):
                model_type = SklearnModelType.POINT_PREDICTION_CLF
                break
            elif isinstance(step, sklearn.base.RegressorMixin):
                model_type = SklearnModelType.REGRESSION
                break
        if not model_type:
            raise SpliceMachineException('Could not determine the type of Pipeline! Model stage is not of '
                                             'classification, regression or clustering.')
        return model_type
            

    @staticmethod
    def get_model_type(model: ScikitModel, sklearn_args: Dict[str, str]) -> SklearnModelType:
        """
        Get the model type of the Scikit-learn model

        :param model: (ScikitModel)
        :param sklearn_args: (Dict[str,str]) the optional scikit-learn args
        :return: SklearnModelType
        """

        # sklearn_args will affect the output type
        if not sklearn_args:
            # Either predict or transform here
            if hasattr(model, 'predict'):
                if isinstance(model, SKPipeline):
                    model_type = SKUtils.get_pipeline_model_type(model)
                elif isinstance(model, sklearn.base.RegressorMixin):
                    model_type = SklearnModelType.REGRESSION
                elif isinstance(model, (sklearn.base.ClusterMixin, sklearn.base.ClassifierMixin)):
                    model_type = SklearnModelType.POINT_PREDICTION_CLF
                else:
                    raise SpliceMachineException(f'Unknown Sklearn Model Type {type(model)}')
            elif hasattr(model, 'transform'): # Transform functions create more than a single point prediction
                model_type = SklearnModelType.KEY_VALUE
            else:
                raise SpliceMachineException(f'Model {type(model)} does not have predict or transform function!')
        else: # If sklearn_args have been passed in, the model must be returning multiple key values
            model_type = SklearnModelType.KEY_VALUE
        return model_type


class KerasUtils:
    @staticmethod
    def log_keras_model(splice_context: PySpliceContext, model: KerasModel, name: str, run_id: str) -> None:
        """
        Logs the Keras Model in the MLManager Aritfacts table

        :param splice_context: (PySpliceContext)
        :param model: (KerasModel)
        :param name: (str) The name to give the model
        :param run_id: (str) The mlflow run id
        :return: None
        """
        model.save('/tmp/model.h5')
        with open('/tmp/model.h5', 'rb') as f:
            byte_stream = bytearray(bytes(f.read()))
        insert_artifact(splice_context, name, byte_stream, run_id, file_ext=FileExtensions.keras)
        remove('/tmp/model.h5')

    @staticmethod
    def load_keras_model(model_blob: object) -> KerasModel:
        """
        Deserializes the serialized Keras Model blob

        :param model_blob: (object) the model blob
        :return: KerasModel
        """
        hfile = h5_file(BytesIO(model_blob), 'r')
        return load_kr_model(hfile)

    @staticmethod
    def insert_keras_model(splice_context: PySpliceContext, run_id: str, model: KerasModel) -> bool:
        """
        Inserts the Keras Model into the MLManager Models table for deployment

        :param splice_context:
        :param run_id:
        :param model:
        :return:
        """
        model.save('/tmp/model.h5')
        with open('/tmp/model.h5', 'rb') as f:
            byte_stream = bytearray(bytes(f.read()))
        remove('/tmp/model.h5')
        return insert_model(splice_context, run_id, byte_stream, 'keras', KERAS_VERSION)

    @staticmethod
    def get_keras_model_type(model: KerasModel, pred_threshold: float) -> KerasModelType:
        """
        Keras models are either Key value returns or "regression" (single value)
        If the output layer has multiple nodes, or there is a threshold (for binary classification), it will
        be a key value return because n values will be returned (n = # output nodes + 1)
        :param model: (KerasModel)
        :param pred_threshold: (float) the prediction threshold if one is passed
        :return: (KerasModelType)
        """

        if model.layers[-1].output_shape[-1] > 1 or pred_threshold:
            model_type = KerasModelType.KEY_VALUE
        else:
            model_type = KerasModelType.REGRESSION
        return model_type

    @staticmethod
    def validate_keras_model(model: KerasModel) -> None:
        """
        Right now, we only support feed forward models. Vector inputs and Vector outputs only
        When we move to LSTMs we will need to change that
        :param model:
        :return:
        """

        input_shape = model.layers[0].input_shape
        output_shape = model.layers[-1].output_shape
        if len(input_shape) != 2 or input_shape[0] or len(output_shape) != 2 or output_shape[0]:
            raise SpliceMachineException("We currently only support feed-forward models. The input and output shapes"
                                         "of the models must be (None, #). Please raise an issue here: https://github.com/splicemachine/pysplice/issues")

    @staticmethod
    def prep_model_for_deployment(splice_context: PySpliceContext,
                                  model: KerasModel,
                                  classes: List[str],
                                  run_id: str,
                                  df: SparkDF or None,
                                  pred_threshold: float or None,
                                  sklearn_args: Dict[str,str] or None)-> (KerasModelType, List[str]):
        """
        Inserts the model into the MODELS table for deployment
        Gets the Keras model type
        Gets the classes (if applicable)
        :param splice_context: PySpliceContext
        :param model: KerasModel
        :param classes: List[str] the class labels
        :param run_id: str
        :param pred_threshold: double
        :return: (KerasModelType, List[str]) the modelType and the classes
        """
        KerasUtils.validate_keras_model(model)
        model_already_exists = KerasUtils.insert_keras_model(splice_context, run_id, model)
        model_type: KerasModelType = KerasUtils.get_keras_model_type(model, pred_threshold)
        if model_type == KerasModelType.KEY_VALUE:
            output_shape = model.layers[-1].output_shape
            if classes and len(classes) != output_shape[-1]:
                raise SpliceMachineException(f'The number of classes, {len(classes)}, does not match '
                                             f'the output shape of your model, {output_shape}')
            elif not classes:
                classes = ['prediction'] + [f'output{i}' for i in range(output_shape[-1])]
            else:
                classes = ['prediction'] + classes
            if len(classes) > 2 and pred_threshold:
                print(f"Found multiclass model with pred_threshold {pred_threshold}. Ignoring threshold.")
        return model_type, classes, model_already_exists



class SparkUtils:
    MODEL_MODULES = [spark_classification.__name__, spark_recommendation.__name__, spark_clustering.__name__,
                       spark_regression.__name__]
    @staticmethod
    def get_stages(pipeline: PipelineModel):
        """
        Extract the stages from a fit or unfit pipeline

        :param pipeline: a fit or unfit Spark pipeline
        :return: stages list
        """
        if hasattr(pipeline, 'getStages'):
            return pipeline.getStages()  # unfit pipeline
        return pipeline.stages  # fit pipeline

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
        Get a readable version of the Pipeline stage (without the memory address)

        :param pipeline_stage: the name of the stage to parse
        """
        if '_' in str(pipeline_stage):
            return str(pipeline_stage).split('_')[0]
        return str(pipeline_stage)

    @staticmethod
    def is_spark_pipeline(spark_object):
        """
        Returns whether or not the given object is a spark pipeline. If it is a Pipeline, it will return True, if it is a
        model is will return False. Otherwise, it will throw an exception

        :param spark_object: (Model) Spark object to check
        :return: (bool) whether or not the object is a model
        :exception: (SpliceMachineException) throws an error if it is not either
        """
        if isinstance(spark_object, PipelineModel):
            return True

        if isinstance(spark_object, SparkModel):
            return False

        raise SpliceMachineException("The model supplied does not appear to be a Spark Model!")

    @staticmethod
    def get_model_stage(pipeline: PipelineModel) -> SparkModel:
        """"
        Gets the Model stage of a FIT PipelineModel

        :param pipeline: (PipelineModel)
        :return: SparkModel
        """
        for i in SparkUtils.get_stages(pipeline):
            # StandardScaler is also implemented as a base Model and JavaModel for some reason but that's not a model
            # So we need to make sure the stage isn't a feature
            if getattr(i, '__module__', None) in SparkUtils.MODEL_MODULES:
                return i
        raise AttributeError("It looks like you're trying to deploy a pipeline without a supported Spark Model. Supported Spark models "
                             "are listed here: https://mleap-docs.combust.ml/core-concepts/transformers/support.html")

    @staticmethod
    def try_get_class_labels(pipeline: PipelineModel):
        """
        Tries to get the class labels for a Spark Model. This will only work if the Pipeline has a Model and an IndexToString
        where the inputCol of the IndexToString is the same as the predictionCol of the Model

        :param pipeline: (PipelineModel) The fitted spark PipelineModel
        :return: List[str] the class labels in order of numeric value (0,1,2 etc)
        """
        labels = []
        if hasattr(pipeline, 'getStages'):
            raise SpliceMachineException('The passed in pipeline has not been fit. Please pass in a fit pipeline')
        model = SparkUtils.get_model_stage(pipeline)
        for stage in SparkUtils.get_stages(pipeline):
            if isinstance(stage, IndexToString):  # It's an IndexToString
                if stage.getOrDefault('inputCol') == model.getOrDefault(
                        'predictionCol'):  # It's the correct IndexToString
                    labels = stage.getOrDefault('labels')
        return labels

    @staticmethod
    def parse_string_parameters(string_parameters: str) -> Dict[str,str]:
        """
        Parse string rendering of extractParamMap

        :param string_parameters: (str) the string parameters
        :return: Dict the parsed parameters
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
    def retrieve_artifact_stream(splice_context: PySpliceContext, run_id: str, name: str):
        """
        Retrieve the binary stream for a given artifact with the specified name and run id

        :param splice_context: (PySpliceContext) the PySpliceContext
        :param run_id: (str) the run id for the run that the artifact belongs to
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
    def get_feature_vector_columns(fittedPipe: PipelineModel) -> List[str]:
        """
        Gets the input columns from the VectorAssembler stage of a fitted Pipeline

        :param fittedPipe: (PipelineModel) The trained model
        :return: List[str] the feature vector columns
        """
        vec_stage = None
        for i in fittedPipe.stages:
            if 'VectorAssembler' in str(i):
                vec_stage = i
                break
        return SparkUtils.get_cols(vec_stage)

    @staticmethod
    def get_num_classes(pipeline_or_model: SparkModel or PipelineModel):
        """
        Tries to find the number of classes in a Pipeline or Model object

        :param pipeline_or_model: The Pipeline or Model object
        """
        if 'pipeline' in str(pipeline_or_model.__class__).lower():
            model = SparkUtils.get_model_stage(pipeline_or_model)
        else:
            model = pipeline_or_model
        num_classes = model.numClasses if model.hasParam('numClasses') else model.numClasses if hasattr(model,
                                                                                                        'numClasses') else model.summary.k
        return num_classes

    @staticmethod
    def get_model_type(pipeline_or_model: SparkModel or PipelineModel) -> SparkModelType:
        """
        Takes a fitted Spark Pipeline or Model and determines if it is a Regression, Classification, or Clustering model

        :param pipeline_or_model: (SparkModel or PipelineModel) the fitted Spark model
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
    def log_spark_model(splice_context: PySpliceContext, model: SparkModel, name: str, run_id: str) -> None:
        """
        Inserts a Serialized spark model into the database as an artifact

        :param splice_context: (PySpliceContext) The PySpliceContext
        :param model: (SparkModel) the trained Spark model
        :param name: (str) the name to give the saved model
        :param run_id: (str) the run id associated to associate to this model
        :return: (None)
        """
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
    def load_spark_model(splice_ctx: PySpliceContext, spark_pipeline_blob: object) -> SparkModel:
        """
        Loads a Serialized Spark Model into a Spark Pipeline

        :param splice_ctx: (PySpliceContext) The PySpliceContext
        :param spark_pipeline_blob: (object) The serialized model object
        :return: (SparkModel) The deserialized Spark model pipeline
        """
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
    def prep_model_for_deployment(splice_context: PySpliceContext,
                                  fittedPipe: PipelineModel,
                                  classes: List[str],
                                  run_id: str,
                                  df: SparkDF,
                                  pred_threshold: float or None,
                                  sklearn_args: Dict[str,str] or None) -> (SparkModelType, List[str]):
        """
        All preprocessing steps to prepare for in DB deployment. Get the mleap model, get class labels

        :param fittedPipe:
        :param df:
        :param classes:
        :return:
        """

        # Check if model is not a pipeline. This would occur when user logs a Pipeline with 1 stage
        if not SparkUtils.is_spark_pipeline(fittedPipe):
            print('You are deploying a singular Spark Model. It will be deployed as a Pipeline with 1 stage. This will'
                  'not affect expected behavior or outcomes.')
            fittedPipe = PipelineModel(stages=[fittedPipe])
        # Get model type
        model_type = SparkUtils.get_model_type(fittedPipe)
        # See if the labels are in an IndexToString stage. Will either return List[str] or empty []
        potential_clases = SparkUtils.try_get_class_labels(fittedPipe)
        classes = classes if classes else potential_clases
        if classes:
            if model_type not in (SparkModelType.CLASSIFICATION, SparkModelType.CLUSTERING_WITH_PROB):
                print('Prediction labels found but model is not type Classification. Removing labels')
                classes = None
            else:
                # handling spaces in class names
                classes = [c.replace(' ', '_') for c in classes]
                print(
                    f'Prediction labels found. Using {classes} as labels for predictions {list(range(0, len(classes)))} respectively')
        else:
            if model_type in (SparkModelType.CLASSIFICATION, SparkModelType.CLUSTERING_WITH_PROB):
                # Add a column for each class of the prediction to output the probability of the prediction
                classes = [f'C{i}' for i in range(SparkUtils.get_num_classes(fittedPipe))]
        # See if the df passed in has already been transformed.
        # If not, transform it
        if 'prediction' not in df.columns:
            df = fittedPipe.transform(df)
        # Get the Mleap model and insert it into the MODELS table
        mleap_model = get_mleap_model(splice_context, fittedPipe, df, run_id)
        model_already_exists = insert_mleap_model(splice_context, run_id, mleap_model)

        return model_type, classes, model_already_exists


def get_model_library(model: object) -> DBLibraries:
    """
    Gets the model library of a trained model

    :param model: The trained model
    :return: DBLibraries
    """
    if isinstance(model, H2OModel):
        lib = DBLibraries.H2OMOJO
    elif isinstance(model, SparkModel):
        lib = DBLibraries.MLeap
    elif isinstance(model, ScikitModel):
        lib = DBLibraries.SKLearn
    elif isinstance(model, KerasModel):
        lib = DBLibraries.Keras
    else:
        raise SpliceMachineException(f"Submitted Model is not valid for database deployment! Valid models are {DBLibraries.SUPPORTED_LIBRARIES}")
    return lib


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
    Get the current logged in user to Jupyter

    :return: (str) name of the logged in user
    """
    try:
        uname = env_vars.get('JUPYTERHUB_USER') or env_vars['USER']
        return uname
    except KeyError:
        raise Exception(
            "Could not determine current running user. Running MLManager outside of Splice Machine"
            " Cloud Jupyter is currently unsupported")


def insert_model(splice_context: PySpliceContext, run_id: str, byte_array: bytearray, library: str, version: str) -> bool:
    """
    Insert a serialized model into the Mlmanager models table for in database model deployment

    :param splice_context: (PySpliceContext) The PySpliceContext
    :param run_id: (str) mlflow run id
    :param byte_array: (bytearray) byte array
    :param library: (str) The library of the model (mleap, h2omojo etc)
    :param version: (str) The version of the library
    :return: bool whether or not the model already exists
    """
    # If a model with this run_id already exists in the table, gracefully fail
    # May be faster to use prepared statement
    model_exists = splice_context.df(
        f'select count(*) from {SQL.MLMANAGER_SCHEMA}.models where RUN_UUID=\'{run_id}\'').collect()[0][0]
    if model_exists:
        print(
            'A model with this run ID is already deployed. We are NOT replacing it. We will use the currently existing model.\nTo replace, use a new run_id')
        return True

    else:
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
        return False


def insert_artifact(splice_context: PySpliceContext,
                    name: str,
                    byte_array: bytearray,
                    run_uuid: str,
                    file_ext: str=None):
    """
    Insert a serialized object into the Mlmanager artifacts table

    :param splice_context: (PySpliceContext) the PySpliceContext
    :param name: (str) the path to store the binary under (with respect to the current run)
    :param byte_array: (bytearray) Java byte array
    :param run_uuid: (str) run uuid for the run
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


def get_pod_uri(pod: str, port: str or int, _testing: bool=False):
    """
    Get address of MLFlow Container for Kubernetes

    :param pod: (str) the url of the mlflow pod
    :param port: (str or int) the port of the pod for mlflow communication
    :param _testing: (bool) Whether you are testing [default False]
    :return: (str) the pod URI
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


def get_mleap_model(splice_context: PySpliceContext,
                    fittedPipe:PipelineModel,
                    df: SparkDF,
                    run_id: str) -> object:
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

    try:
        fittedPipe.serializeToBundle(f"jar:file:///tmp/{run_id}.zip", df)
    except:
        m = getattr(fittedPipe, '__class__', 'UnknownModel')
        raise SpliceMachineException(f'It look like your model type {m} is not supported. Supported models are listed'
                                     f'here https://mleap-docs.combust.ml/core-concepts/transformers/support.html') from None

    jvm = splice_context.jvm
    java_import(jvm, "com.splicemachine.mlrunner.FileRetriever")
    obj = jvm.FileRetriever.loadBundle(f'jar:file:///tmp/{run_id}.zip')
    remove(f'/tmp/{run_id}.zip')
    return obj


def insert_mleap_model(splice_context: PySpliceContext,
                       run_id: str,
                       model: PipelineModel or SparkModel) -> bool:
    """
    Insert an MLeap Transformer model into the database as a Blob

    :param splice_context: pysplicectx
    :param model_id: (str) the mlflow run_id that the model is associated with
    (with respect to the current run)
    :param model: (PipelineModel) the fitted Mleap Transformer (pipeline)
    :return: (bool) whether or not the model exists already
    """

    # Serialize Mleap model to BLOB
    baos = splice_context.jvm.java.io.ByteArrayOutputStream()
    oos = splice_context.jvm.java.io.ObjectOutputStream(baos)
    oos.writeObject(model)
    oos.flush()
    oos.close()
    byte_array = baos.toByteArray()
    return insert_model(splice_context, run_id, byte_array, 'mleap', MLEAP_VERSION)


def validate_primary_key(splice_ctx: PySpliceContext,
                         primary_key: List[Tuple[str,str]] or None,
                         schema: str or None,
                         table: str or None) -> List[str] or None:
    """
    Validates the primary key passed by the user conforms to SQL. If the user is deploying to an existing table
    This verifies that the table has a primary key

    :param splice_ctx: (PySpliceContext) The PySpliceContext
    :param primary_key: (List[Tuple[str,str]]) column name, SQL datatype for the primary key(s) of the table
    :param schema: (str) the name of the schema being deployed to
    :param table (str) the name of the table being deployed to
    :return: (List[str]) the primary keys
    """
    if primary_key:
        regex = re.compile('[^a-zA-Z]')
        for i in primary_key:
            sql_datatype = regex.sub('', i[1]).upper()
            if sql_datatype not in SQL_TYPES:
                raise ValueError(f'Primary key parameter {i} does not conform to SQL type.'
                                 f'Value {i[1]} should be a SQL type but isn\'t')
    else:
        ps = splice_ctx.getConnection().prepareStatement(f"CALL sysibm.sqlprimarykeys('splicedb','{schema}','{table}','')")
        rs = ps.executeQuery()
        pks = []
        while rs.next(): pks.append(rs.getString(4).lower())
        if not pks:
            raise SpliceMachineException('The provided table has no primary keys. It must have a primary key before deployment')
        pks = [(i,None) for i in pks]
        return pks


def create_model_deployment_table(splice_context: PySpliceContext,
                                  run_id: str,
                                  schema_table_name: str,
                                  schema_str: str,
                                  classes: List[str],
                                  primary_key: List[Tuple[str,str]],
                                  model_type: Enum,
                                  verbose: bool) -> None:
    """
    Creates the table that holds the columns of the feature vector as well as a unique MOMENT_ID

    :param splice_context: pysplicectx
    :param run_id: (str) the run_id for this model
    :param schema_table_name: (str) the schema.table to create the table under
    :param schema_str: (str) the structure of the schema of the table as a string (col_name TYPE,)
    :param classes: (List[str]) the labels of the model (if they exist)
    :param primary_key: List[Tuple[str,str]] column name, SQL datatype for the primary key(s) of the table
    :param model_type: (ModelType) Whether the model is a Regression, Classification or Clustering (with/without probabilities)
    :param verbose: (bool) whether to print the SQL query
    :return: None
    """
    if splice_context.tableExists(schema_table_name):
        raise SpliceMachineException(
            f'The table {schema_table_name} already exists. To deploy to an existing table, do not pass in a dataframe or '
            f'The create_model_table parameter')
    SQL_TABLE = f"""CREATE TABLE {schema_table_name} (\
                \tCUR_USER VARCHAR(50) DEFAULT CURRENT_USER,
                \tEVAL_TIME TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                \tRUN_ID VARCHAR(50) DEFAULT '{run_id}',
                \n {schema_str}
                """

    pk_cols = ''
    for i in primary_key:
        # If pk is already in the schema_string, don't add another column. PK may be an existing value
        if i[0] not in schema_str:
            SQL_TABLE += f'\t{i[0]} {i[1]},\n'
        pk_cols += f'{i[0]},'


    if model_type in (SparkModelType.REGRESSION, H2OModelType.REGRESSION, SklearnModelType.REGRESSION,
                      KerasModelType.REGRESSION):
        SQL_TABLE += '\tPREDICTION DOUBLE,\n'

    elif model_type in (SparkModelType.CLASSIFICATION, SparkModelType.CLUSTERING_WITH_PROB,
                       H2OModelType.CLASSIFICATION):
        SQL_TABLE += '\tPREDICTION VARCHAR(5000),\n'
        for i in classes:
            SQL_TABLE += f'\t"{i}" DOUBLE,\n'
    elif model_type in (H2OModelType.KEY_VALUE, SklearnModelType.KEY_VALUE, KerasModelType.KEY_VALUE):
        for i in classes:
            SQL_TABLE += f'\t"{i}" DOUBLE,\n'

    elif model_type in (SparkModelType.CLUSTERING_WO_PROB, H2OModelType.SINGULAR, SklearnModelType.POINT_PREDICTION_CLF):
        SQL_TABLE += '\tPREDICTION INT,\n'

    SQL_TABLE += f'\tPRIMARY KEY({pk_cols.rstrip(",")})\n)'

    if verbose: print('\n', SQL_TABLE, end='\n\n')
    splice_context.execute(SQL_TABLE)

def alter_model_table(splice_context: PySpliceContext,
                      run_id: str,
                      schema_table_name: str,
                      classes: List[str],
                      model_type: Enum,
                      verbose: bool) -> None:
    """
    Alters the provided table for deployment. Adds columns for storing model results as well as metadata such as
    current user, eval time, run_id, and the prediction label columns

    :param splice_context: pysplicectx
    :param run_id: (str) the run_id for this model
    :param schema_table_name: (str) the schema.table to create the table under
    :param classes: (List[str]) the labels of the model (if they exist)
    :param primary_key: List[Tuple[str,str]] column name, SQL datatype for the primary key(s) of the table
    :param model_type: (ModelType) Whether the model is a Regression, Classification or Clustering (with/without probabilities)
    :param verbose: (bool) whether to print the SQL query
    :return: None
    """

    # Table needs to exist
    if not splice_context.tableExists(schema_table_name):
        raise SpliceMachineException(
            f'The table {schema_table_name} does not exist. To create a new table for deployment, pass in a dataframe and '
            f'The set create_model_table=True')

    # Currently we only support deploying 1 model to a table
    schema = splice_context.getSchema(schema_table_name)
    reserved_fields = set(['CUR_USER', 'EVAL_TIME', 'RUN_ID', 'PREDICTION'] + classes)
    for field in schema:
        if field.name in reserved_fields:
            raise SpliceMachineException(f'The table {schema_table_name} looks like it already has values associated with '
                                         f'a deployed model. Only 1 model can be deployed to a table currently.'
                                         f'The table cannot have the following fields: {reserved_fields}')

    # Splice cannot currently add multiple columns in an alter statement so we need to make a bunch and execute all of them
    SQL_ALTER_TABLE = []
    alter_table_syntax = f'ALTER TABLE {schema_table_name} ADD COLUMN'
    SQL_ALTER_TABLE.append(f'{alter_table_syntax} CUR_USER VARCHAR(50) DEFAULT CURRENT_USER')
    SQL_ALTER_TABLE.append(f'{alter_table_syntax} EVAL_TIME TIMESTAMP DEFAULT CURRENT_TIMESTAMP')
    SQL_ALTER_TABLE.append(f'{alter_table_syntax} RUN_ID VARCHAR(50) DEFAULT \'{run_id}\'')

    # Add the correct prediction type
    if model_type in (SparkModelType.REGRESSION, H2OModelType.REGRESSION, SklearnModelType.REGRESSION,
                      KerasModelType.REGRESSION):
        SQL_ALTER_TABLE.append(f'{alter_table_syntax} PREDICTION DOUBLE')

    elif model_type in (SparkModelType.CLASSIFICATION, SparkModelType.CLUSTERING_WITH_PROB,
                       H2OModelType.CLASSIFICATION):
        SQL_ALTER_TABLE.append(f'{alter_table_syntax} PREDICTION VARCHAR(5000)')
        for i in classes:
            SQL_ALTER_TABLE.append(f'{alter_table_syntax} "{i}" DOUBLE')

    elif model_type in (H2OModelType.KEY_VALUE, SklearnModelType.KEY_VALUE, KerasModelType.KEY_VALUE):
        for i in classes:
            SQL_ALTER_TABLE.append(f'{alter_table_syntax} "{i}" DOUBLE')

    elif model_type in (SparkModelType.CLUSTERING_WO_PROB, H2OModelType.SINGULAR, SklearnModelType.POINT_PREDICTION_CLF):
        SQL_ALTER_TABLE.append(f'{alter_table_syntax} PREDICTION INT')

    # SQL_TABLE += f'\tPRIMARY KEY({pk_cols.rstrip(",")})\n)'
    for sql in SQL_ALTER_TABLE:
        if verbose: print(sql)
        splice_context.execute(sql)

def create_vti_prediction_trigger(splice_context: PySpliceContext,
                                  schema_table_name: str,
                                  run_id: str,
                                  feature_columns: List[str],
                                  schema_types: Dict[str,str],
                                  schema_str: str,
                                  primary_key: List[Tuple[str, str]],
                                  classes: List[str],
                                  model_type: Enum,
                                  sklearn_args: Dict[str, str],
                                  pred_threshold: float,
                                  verbose: bool) -> None:
    """
    Creates the VTI trigger for model types that use VTIs instead of standard Java Functions

    :param splice_context: (PySpliceContext) the PySpliceContext
    :param schema_table_name: (str) the schema.table to create the table under
    :param run_id: (str) the run_id to deploy the model under
    :param feature_columns: (List[str]) the original features that are transformed into the final feature vector
    :param schema_types: (Dict[str, str]) a mapping of feature column to data type
    :param schema_str: (str) the structure of the schema of the table as a string (col_name TYPE,)
    :param primary_key: List[Tuple[str,str]] column name, SQL datatype for the primary key(s) of the table
    :param classes: (List[str]) the label columns of the model prediction
    :param model_type: (Enum) Whether the model is a Regression, Classification or Clustering (with/without probabilities)
    :param sklearn_args: (Dict[str,str]) Any custom scikit-learn prediction arguments
    :param pred_threshold: (float) the optional keras prediction threshold for predictions
    :param verbose: (bool) whether to print the SQL query
    :return: None
    """

    prediction_call = "new com.splicemachine.mlrunner.MLRunner('key_value', '{run_id}', {raw_data}, '{schema_str}'"

    if model_type == SklearnModelType.KEY_VALUE:
        if not sklearn_args: # This must be a .transform call
            predict_call, predict_args = 'transform', None
        elif 'predict_call' in sklearn_args and 'predict_args' in sklearn_args:
            predict_call, predict_args = sklearn_args['predict_call'], sklearn_args['predict_args']
        elif 'predict_args' in sklearn_args:
            predict_call, predict_args = 'predict', sklearn_args['predict_args']
        elif 'predict_call' in sklearn_args:
            predict_call, predict_args = sklearn_args['predict_call'], None
        else:
            raise SpliceMachineException('You had an invalid key in your sklearn_args. Valid keys (predict_call, predict_args)')

        prediction_call += f", '{predict_call}', '{predict_args}'"

    elif model_type == KerasModelType.KEY_VALUE and len(classes) == 2 and pred_threshold:
        prediction_call +=  f", '{pred_threshold}'"

    prediction_call += ')'
    schema = schema_table_name.split('.')[0]
    SQL_PRED_TRIGGER = f'CREATE TRIGGER {schema}.runModel_{schema_table_name.replace(".", "_")}_{run_id}\n \tAFTER INSERT\n ' \
                       f'\tON {schema_table_name}\n \tREFERENCING NEW AS NEWROW\n \tFOR EACH ROW\n \t\tUPDATE ' \
                       f'{schema_table_name} SET ('

    output_column_names = ''  # Names of the output columns from the model
    output_cols_VTI_reference = ''  # Names references from the VTI (ie b.COL_NAME)
    output_cols_schema = ''  # Names with their datatypes (always DOUBLE)
    for i in classes:
        output_column_names += f'"{i}",'
        output_cols_VTI_reference += f'b."{i}",'
        output_cols_schema += f'"{i}" DOUBLE,' if i != 'prediction' else f'"{i}" INT,' #for sklearn predict_proba

    raw_data = ''
    for i, col in enumerate(feature_columns):
        raw_data += '||' if i != 0 else ''
        if schema_types[str(col)] == 'StringType':
            raw_data += f'NEWROW.{col}||\',\''
        else:
            inner_cast = f'CAST(NEWROW.{col} as DECIMAL(38,10))' if schema_types[str(col)] in {'FloatType', 'DoubleType',
                                                                                           'DecimalType'} else f'NEWROW.{col}'
            raw_data += f'TRIM(CAST({inner_cast} as CHAR(41)))||\',\''

    # Cleanup + schema for PREDICT call
    raw_data = raw_data[:-5].lstrip('||')
    schema_str_pred_call = schema_str.replace('\t', '').replace('\n', '').rstrip(',')

    prediction_call = prediction_call.format(run_id=run_id, raw_data=raw_data, schema_str=schema_str_pred_call)


    SQL_PRED_TRIGGER += f'{output_column_names[:-1]}) = ('
    SQL_PRED_TRIGGER += f'SELECT {output_cols_VTI_reference[:-1]} FROM {prediction_call}' \
                        f' as b ({output_cols_schema[:-1]}) WHERE 1=1) WHERE '

    for i in primary_key:
        SQL_PRED_TRIGGER += f'{i[0]} = NEWROW.{i[0]} AND'
    # Remove last AND
    SQL_PRED_TRIGGER = SQL_PRED_TRIGGER[:-3]

    if verbose:
        print()
        print(SQL_PRED_TRIGGER, end='\n\n')
    splice_context.execute(SQL_PRED_TRIGGER.replace('\t', ' '))


def create_prediction_trigger(splice_context: PySpliceContext,
                              schema_table_name: str,
                              run_id: str,
                              feature_columns: List[str],
                              schema_types: Dict[str,str],
                              schema_str: str,
                              primary_key: str,
                              model_type: Enum,
                              verbose: bool) -> None:
    """
    Creates the trigger that calls the model on data row insert. This trigger will call predict when a new row is inserted into the data table
    and update the row to contain the prediction(s)

    :param splice_context: (PySpliceContext) the PySpliceContext
    :param schema_table_name: (str) the schema.table to create the table under
    :param run_id: (str) the run_id to deploy the model under
    :param feature_columns: (List[str]) the original features that are transformed into the final feature vector
    :param schema_types: (Dict[str, str]) a mapping of feature column to data type
    :param schema_str: (str) the structure of the schema of the table as a string (col_name TYPE,)
    :param primary_key: List[Tuple[str,str]] column name, SQL datatype for the primary key(s) of the table
    :param model_type: (Enum) Whether the model is a Regression, Classification or Clustering (with/without probabilities)
    :param verbose: (bool) whether to print the SQL query
    :return: None
    """

    if model_type in (SparkModelType.CLASSIFICATION, H2OModelType.CLASSIFICATION):
        prediction_call = 'MLMANAGER.PREDICT_CLASSIFICATION'
    elif model_type in (SparkModelType.REGRESSION, H2OModelType.REGRESSION,
                        SklearnModelType.REGRESSION, KerasModelType.REGRESSION):
        prediction_call = 'MLMANAGER.PREDICT_REGRESSION'
    elif model_type == SparkModelType.CLUSTERING_WITH_PROB:
        prediction_call = 'MLMANAGER.PREDICT_CLUSTER_PROBABILITIES'
    elif model_type in (SparkModelType.CLUSTERING_WO_PROB, H2OModelType.SINGULAR, SklearnModelType.POINT_PREDICTION_CLF):
        prediction_call = 'MLMANAGER.PREDICT_CLUSTER'
    elif model_type == H2OModelType.KEY_VALUE:
        prediction_call = 'MLMANAGER.PREDICT_KEY_VALUE'

    schema = schema_table_name.split('.')[0]
    SQL_PRED_TRIGGER = f'CREATE TRIGGER {schema}.runModel_{schema_table_name.replace(".", "_")}_{run_id}\n \tBEFORE INSERT\n ' \
                       f'\tON {schema_table_name}\n \tREFERENCING NEW AS NEWROW\n \tFOR EACH ROW\n \tBEGIN ATOMIC \t\t' \
                       f'SET NEWROW.PREDICTION='

    SQL_PRED_TRIGGER += f'{prediction_call}(\'{run_id}\','

    for i, col in enumerate(feature_columns):
        SQL_PRED_TRIGGER += '||' if i != 0 else ''
        if schema_types[str(col)] == 'StringType':
            SQL_PRED_TRIGGER += f'NEWROW.{col}||\',\''
        else:
            inner_cast = f'CAST(NEWROW.{col} as DECIMAL(38,10))' if schema_types[str(col)] in {'FloatType', 'DoubleType',
                                                                                               'DecimalType'} else f'NEWROW.{col}'
            SQL_PRED_TRIGGER += f'TRIM(CAST({inner_cast} as CHAR(41)))||\',\''

    # Cleanup + schema for PREDICT call
    SQL_PRED_TRIGGER = SQL_PRED_TRIGGER[:-5].lstrip('||') + ',\n\'' + schema_str.replace('\t', '').replace('\n',
                                                                                                           '').rstrip(
        ',') + '\');END'
    if verbose:
        print()
        print(SQL_PRED_TRIGGER, end='\n\n')
    splice_context.execute(SQL_PRED_TRIGGER.replace('\n', ' ').replace('\t', ' '))


def create_parsing_trigger(splice_context: PySpliceContext, schema_table_name: str, primary_key: str,
                           run_id: str, classes: List[str], model_type: Enum, verbose: bool) -> None:
    """
    Creates the secondary trigger that parses the results of the first trigger and updates the prediction row populating the relevant columns

    :param splice_context: (PySpliceContext) the PySpliceContext
    :param schema_table_name: (str) the schema.table to create the table under
    :param primary_key: List[Tuple[str,str]] column name, SQL datatype for the primary key(s) of the table
    :param run_id: (str) the run_id to deploy the model under
    :param classes: (List[str]) the labels of the model (if they exist)
    :param model_type: (Enum) the model type (H2OModelType or SparkModelType)
    :param verbose: (bool) whether to print the SQL quer
    :return: None
    """
    schema = schema_table_name.split('.')[0]
    SQL_PARSE_TRIGGER = f'CREATE TRIGGER {schema}.PARSERESULT_{schema_table_name.replace(".", "_")}_{run_id}' \
                        f'\n \tBEFORE INSERT\n \tON {schema_table_name}\n \tREFERENCING NEW AS NEWROW\n' \
                        f' \tFOR EACH ROW\n \t\tBEGIN ATOMIC\n\t set '
    set_prediction_case_str = 'NEWROW.PREDICTION=\n\t\tCASE\n'
    for i, c in enumerate(classes):
        SQL_PARSE_TRIGGER += f'NEWROW."{c}"=MLMANAGER.PARSEPROBS(NEWROW.prediction,{i}),'
        set_prediction_case_str += f'\t\tWHEN MLMANAGER.GETPREDICTION(NEWROW.prediction)={i} then \'{c}\'\n'

    set_prediction_case_str += '\t\tEND;'
    if model_type == H2OModelType.KEY_VALUE:  # These models don't have an actual prediction
        SQL_PARSE_TRIGGER = SQL_PARSE_TRIGGER[:-1] + 'END'
    else:
        SQL_PARSE_TRIGGER += set_prediction_case_str + 'END'

    if verbose:
        print()
        print(SQL_PARSE_TRIGGER, end='\n\n')
    splice_context.execute(SQL_PARSE_TRIGGER.replace('\n', ' ').replace('\t', ' '))

def get_feature_columns_and_types(splice_ctx: PySpliceContext,
                                   df: SparkDF or None,
                                   create_model_table: bool,
                                   model_cols: List[str] or None,
                                   schema_table_name: str) -> Tuple[List[str], Dict[str,str]]:
    """
    Gets the features and their data types for the table of the deployed model

    :param df: The dataframe or None
    :param create_model_table: bool if the user wants to create the table
    :param schema_table_name: The table in question
    :param model_cols: List[str] the columns that go into the feature vector for the model or None
    :return: (Tuple[List[str], Dict[str,str]]) The ordered feature columns as well as the schema types
    """
    if create_model_table:
        assert type(df) in (SparkDF, PandasDF), "Dataframe must be a PySpark or Pandas dataframe!"
        if type(df) == PandasDF:
            df = splice_ctx.spark_session.createDataFrame(df)
        feature_columns = df.columns
        # Get the datatype of each column in the dataframe
        schema_types = {str(i.name): re.sub("[0-9,()]", "", str(i.dataType)) for i in df.schema}
    # Else they are deploying to an existing table
    else:
        if not splice_ctx.tableExists(schema_table_name):
            raise SpliceMachineException("You've tried to deploy a model to a table that does not exist. "
                                         "If you'd like to create the table using this function, please pass in a dataframe"
                                         "and set create_model_table=True")
        schema_from_table = splice_ctx.getSchema(schema_table_name)
        if model_cols:
            m = set([c.upper() for c in model_cols]) # set is O(1) lookup
            schema_types = {str(i.name): re.sub("[0-9,()]", "", str(i.dataType)) for i in schema_from_table if i.name.upper() in m}
            feature_columns = [i.name for i in schema_from_table if i.name.upper() in m]
        else:
            feature_columns = [i.name for i in schema_from_table]
            schema_types = {str(i.name): re.sub("[0-9,()]", "", str(i.dataType)) for i in schema_from_table}

    return feature_columns, schema_types

def get_df_for_mleap(splice_ctx: PySpliceContext,
                      schema_table_name: str,
                      df: SparkDF or PandasDF or None) -> SparkDF:
    """
    Get the dataframe for the deployment if the deployment is of a Spark model
    MLeap needs a dataframe in order to serialize the model. If it's not passed in, we need to get it from an existing table

    :param splice_ctx: PySpliceContext
    :param schema_table_name: str the table to get the dataframe from
    :param df: the dataframe if it exists or none
    :param create_model_table: bool if the user wants to create the table
    :return: (SparkDF) the dataframe
    """
    if df:
        if type(df) == PandasDF:
            df = splice_ctx.spark_session.createDataFrame(df)

    elif not splice_ctx.tableExists(schema_table_name):
        raise SpliceMachineException('MLeap requires a dataframe to serialize a spark model. You must either pass in a '
                                     'dataframe to use for serialization or provide a table that already exists into this function.'
                                     'Note that the model will be deployed to that table.')
    else:
        df = splice_ctx.df(f'select top 1 * from {schema_table_name}')

    return df

def add_model_to_metadata(splice_context: PySpliceContext,
                          run_id: str,
                          schema_table_name: str) -> None:
    """
    Adds the deployed model to the model_metadata table

    :param splice_context: (PySpliceContext) the PySpliceContext
    :param run_id: (str) the run_id of the deployment
    :param schema_table_name: (str) the SCHEMA.TABLE that the model was deployed to
    :return: (None)
    """

    if splice_context.tableExists(f'{SQL.MLMANAGER_SCHEMA}.MODEL_METADATA'):
        schema_table_name = schema_table_name.upper()
        schema, table = schema_table_name.split('.')

        table_id = splice_context.df(f"select a.tableid from sys.systables a join sys.sysschemas b on a.schemaid=b.schemaid "
                                          f"where a.tablename='{table}' and b.schemaname='{schema}'").collect()[0][0]

        trigger_name_1 = f"RUNMODEL_{schema_table_name.replace('.','_')}_{run_id}".upper()
        trigger_id_1, create_ts = splice_context.df(f"select triggerid, varchar(creationtimestamp) from sys.systriggers "
                                                    f"where triggername='{trigger_name_1}' and tableid='{table_id}'")\
                                                    .collect()[0]

        # Not all models will have a second trigger
        trigger_name_2 = f"PARSERESULT_{schema_table_name.replace('.', '_')}_{run_id}".upper()
        trigger_id_2 = splice_context.df(f"select triggerid from sys.systriggers where triggername='{trigger_name_2}' "
                                         f"and tableid='{table_id}'").collect()

        # Adding extra single quote to trigger_id_2  case NULL
        trigger_id_2 = f"'{trigger_id_2[0][0]}'" if trigger_id_2 else 'NULL'

        # We don't add the quotes around trigger_id_2 here because we handle it above in the NULL case
        splice_context.execute(f"INSERT INTO {SQL.MLMANAGER_SCHEMA}.MODEL_METADATA"
                               f"(RUN_UUID, ACTION, TABLEID, TRIGGER_TYPE, TRIGGERID, TRIGGERID_2, DB_ENV, DB_USER, ACTION_DATE)"
                               f"values ('{run_id}', 'DEPLOYED', '{table_id}', 'INSERT', '{trigger_id_1}', {trigger_id_2},"
                               f"'PROD', '{get_user()}', '{create_ts}')")



def drop_tables_on_failure(splice_context: PySpliceContext,
                           schema_table_name: str,
                           run_id: str,
                           model_already_exists: bool) -> None:
    """
    Attempts to roll back transactions from the deploy_db function on failure

    Due to some limitations DB-7726 we can't use fully utilize a single consistent JDBC connection using NSDS
    So we will try to rollback on failure using basic logic.

    If the model was already in the models table (ie it had been deployed before), we will leave it. Otherwise, delete
    Leave the tables.
    """

    # splice_context.execute(f'DROP TABLE IF EXISTS {schema_table_name}')
    if not model_already_exists:
        splice_context.execute(f'DELETE FROM {SQL.MLMANAGER_SCHEMA}.MODELS WHERE RUN_UUID=\'{run_id}\'')

ModelUtils = {
    DBLibraries.MLeap: SparkUtils,
    DBLibraries.H2OMOJO: H2OUtils,
    DBLibraries.Keras: KerasUtils,
    DBLibraries.SKLearn: SKUtils,
}
