from typing import List, Dict, Optional
from pyspark.sql.dataframe import DataFrame as SparkDF
from datetime import datetime

class FeatureStore:
    def __init__(self, splice):
        self.splice_ctx = splice

    def get_feature_tables(self) -> List[str]:
        """
        Returns a list of available feature tables

        :return: list[str]
        """
        pass

    def create_feature_table(self, name: str, pk_columns: Dict[str,str], feature_column: Dict[str,str]):
        """
        Creates a new feature table
        :return:
        """
        pass

    def create_training_set(self, *, name: str, sql: str, features: List[str], primary_keys: List[str],
                            ts_col: str, label_col: Optional[str] = None, replace: Optional[bool] = False,
                            desc: Optional[str] = None) -> None:
        """
        Generates and registers a training set

        :param name: The training set name
        :param sql: (str) a SELECT statement that includes:
            * the primary key column(s) - uniquely identifying a training row/case
            * the inference timestamp column - timestamp column with which to join features (temporal join timestamp)
            * context key(s) - the references to the other feature tables' primary keys (ie customer_id, location_id)
            * (optionally) the label expression - defining what the training set is trying to predict
        :param features: (List[str]) the list of features from the feature store to be included in the feature set
            * NOTE: This function will error if the training SQL is missing a context key required to retrieve the\
             desired features
        :param primary_keys: (List[str]) The list of columns from the training SQL that identify the training row
        :param ts_col: (Optional[str]) The timestamp column of the training SQL that identifies the inference timestamp
        :param label_col: (Optional[str]) The optional label column from the training SQL.
        :param replace: (Optional[bool]) Whether to replace an existing training set
        :param desc: (Optional[str]) An optional description of teh training set
        :return: Nothing
        """

        # validate_sql()
        # prepare_training_sql()
        # register_training_sql()
        return self.get_training_set(name)
        pass

    def get_context_keys(self, features: List[str]) -> Dict[str,List[str]]:
        """
        Returns a dictionary mapping each individual feature to its primary key(s)
        :param features: (List[str]) The list of features to get primary keys for
        :return: Dict[str, List[str]]
        """
        pass

    def get_training_set(self, name: str, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None,
                         return_sql=False) -> Optional[SparkDF, str]:
        """
        Returns the training set as a Spark Dataframe

        :param name: (str) The name of the registered training set
        :param start_time: (Optional[datetime]) The start time of the query (how far back in the data to start). Default None
            * NOTE: If start_time is None, query will start from beginning of history
        :param end_time: (Optional[datetime]) The end time of the query (how far recent in the data to get). Default None
            * NOTE: If end_time is None, query will get most recently available data
        :param return_sql: (Optional[bool]) whether to return the dataframe or the SQL that creates it. Defaults False
        :return: Optional[SparkDF, str]
        """
        pass

    def list_training_sets(self) -> Dict[str, Optional[str]]:
        """
        Returns a list a training sets available, and optionally the description if there is one
        :return: Dict[str, Optional[str]]
        """
        pass

