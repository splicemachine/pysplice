from os import environ as env_vars
import os

from sys import getsizeof
from typing import Dict, List

from pyspark.ml.base import Model as SparkModel
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.wrapper import JavaModel
import requests

from ..spark.context import PySpliceContext


class SpliceMachineException(Exception):
    pass


class SparkUtils:
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
            if isinstance(i, SparkModel) and isinstance(i, JavaModel) and 'feature' not in i.__module__:
                return i
        raise AttributeError('Could not find model stage in Pipeline! Is this a fitted spark Pipeline?')

    @staticmethod
    def parse_string_parameters(string_parameters: str) -> Dict[str, str]:
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
    def find_spark_transformer_inputs_by_output(dictionary, value):
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


def get_user():
    """
    Get the current logged in user to Jupyter

    :return: (str) name of the logged in user
    """
    uname = env_vars.get('JUPYTERHUB_USER') or env_vars.get('USER')
    return uname


def get_pod_uri(pod: str, port: str or int, _testing: bool = False):
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

def get_jobs_uri(mlflow_uri: str):
    """
    Returns the Jobs uri given an mlflow URI. The mlflow uri will either be a URL:5001 or
    a cloud url /mlflow depending on where the user is. This handles either and returns the proper uri.

    :param mlflow_uri: The mlflow uri
    :return: The jobs uri
    """
    if mlflow_uri.endswith(':5001'):
        return mlflow_uri.replace(':5001',':5003')
    elif mlflow_uri.endswith('/mlflow'):
        return mlflow_uri + '-jobs'
    else:
        raise SpliceMachineException(f'The provided URI {mlflow_uri} doesn\'t conform for the expected uri.'
                                     f'The Mlflow URI should either end :5001 or /mlflow')

def insert_artifact(host, filepath, name, run_id, file_extension, auth, artifact_path = None):
    """
    Inserts an artifact into the Splice Artifact Store

    :param host: The Splice Artifact store host
    :param filepath: The local path to the file
    :param name: File name
    :param run_id: Run ID
    :param file_extension: File extension
    :param auth: Basic Auth or JWT
    :param artifact_path: Optional artifact directory path. This is for storing directories as artifacts
    """
    payload = dict(
        name = name,
        run_id = run_id,
        file_extension=file_extension,
        artifact_path=artifact_path
    )
    print('Uploading file... ', end='')
    with open(filepath, 'rb') as file:
        r = requests.post(
            host + '/api/rest/upload-artifact',
            auth=auth,
            data=payload,
            files={'file': file}
        )
    if not r.ok:
        raise SpliceMachineException(r.text)
    print('Done.')

def download_artifact(host, name: str, run_id: str, auth) -> requests.models.Response:
    """
    Downloads an artifact from the Splice Machine artifact store

    :param host: The Splice Artifact store host
    :param name: Artifact name. If the artifact was stored with an extension, this name must include that extension.
    :param run_id: The run ID that the artifact is stored under
    :param auth: Basic Auth or JWT
    :return: Http Response
    """
    payload = dict(
        name = name,
        run_id = run_id
    )
    print(f'Downloading file {name}')
    r = requests.get(
        host + '/api/rest/download-artifact',
        params=payload,
        auth=auth
    )
    if not r.ok:
        raise SpliceMachineException(r.text)
    print('Done. Unpacking artifact')
    return r
