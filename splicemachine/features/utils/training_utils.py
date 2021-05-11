from splicemachine import SpliceMachineException
from typing import Dict, Any, Union
from splicemachine.spark import PySpliceContext, ExtPySpliceContext
import pandas as pd
import json

"""
A set of utility functions for creating Training Set SQL 
"""

class ReturnType:
    """
    An enum class for available Training Set return types
    """
    SQL = 'sql'
    SPARK = 'spark'
    PANDAS = 'pandas'
    JSON = 'json'

    @staticmethod
    def get_valid():
        return {ReturnType.SQL, ReturnType.PANDAS, ReturnType.SPARK, ReturnType.JSON}
    @staticmethod
    def map_to_request(rt: str) -> str:
        """
        Maps a users requested type to the proper type for the REST api request
        :param rt: The return type (sql, json, pandas, spark)
        :return: The proper return type for the REST api
        """
        if rt in (ReturnType.PANDAS, ReturnType.JSON):
            return ReturnType.JSON
        elif rt not in ReturnType.get_valid():
            return ''
        return rt


def _format_training_set_output(response: Dict[str, Any], return_type: str,
                                splice_ctx: Union[PySpliceContext, ExtPySpliceContext]):
    """
    Calls to get_training_set, get_training_set_from_view, get_training_set_from_deployment, get_training_set_by_name
    have an optional return_type parameter. This will change what is returned.

    json -> Return the raw response
    pandas -> Convert raw response to pandas dataframe and return that
    spark -> Run the Training Set SQL via the NSDS and return the Spark DF
    sql -> Return the SQL

    :param response: The response from the call to the API
    :return: The training set in the form requested
    """

    if return_type == ReturnType.SQL:
        return response['sql']
    elif return_type == ReturnType.PANDAS:
        try:
            return pd.DataFrame(dict(json.loads(response['data'])))
        except Exception as e:
            raise SpliceMachineException(f"There was an issue converting the response data to a Pandas dataframe. Consider"
                                         f" setting return_type to 'json' to get the raw data, or 'sql' "
                                         f"to the SQL that was generated. Error from pandas: {str(e)} ")
    elif return_type == ReturnType.JSON:
        return response['data']
    else:
        if not splice_ctx:
            raise SpliceMachineException("You either didn't specify a return_type or set it to 'spark', but you have not "
                                         "registered a PySpliceContext or ExtPySpliceContext. Either register one with the"
                                         "register_splice_context function or change the return type to one of "
                                         f"{ReturnType.get_valid() - {'spark'}}")
        return splice_ctx.df(response['sql'])

