from splicemachine.features import FeatureSet
from typing import List, Dict, Optional, Tuple
from pyspark.sql.dataframe import DataFrame as SparkDF
from datetime import datetime

TIMESTAMP_FORMAT = 'yyyy-MM-dd HH:mm:ss'

class FeatureStore:
    def __init__(self, splice):
        self.splice_ctx = splice
        self.feature_tables = []

    def get_feature_sets(self) -> List[FeatureSet]:
        """
        Returns a list of available feature sets

        :return: List[FeatureTable]
        """
        return self.feature_tables

    def get_training_contexts(self) -> Dict[int, Tuple[str,str]]:
        # TODO: Webinar
        """
        Returns all available training contexts in the format of a dictionary mapping
        Context_ID: (context_name, context_description)

        :return:
        """

    def get_training_context_id(self, name) -> int:
        """
        Returns the unique context ID from a name

        :param name:
        :return:
        """


    def add_feature_set(self, ft: FeatureSet):
        """
        Add a feature set

        :param ft:
        :return:
        """
        self.feature_tables.append(ft)

    def create_feature_set(self, schema: str, name: str, pk_columns: Dict[str,str],
                             feature_column: Dict[str,str], desc: Optional[str] = None) -> FeatureSet:
        # TODO: Webinar
        """
        Creates a new feature set, recording metadata and generating the database table with associated triggers

        :param schema:
        :param name:
        :param pk_columns:
        :param feature_column:
        :param desc:
        :return: FeatureTable
        """
        pass


    def create_training_context(self, *, name: str, sql: str, primary_keys: List[str],
                            ts_col: str, label_col: Optional[str] = None, replace: Optional[bool] = False,
                            desc: Optional[str] = None) -> None:
        # TODO: Webinar
        """
        Registers a training context for use in generating training SQL

        :param name: The training set name. This must be unique to other existing training sets unless replace is True
        :param sql: (str) a SELECT statement that includes:
            * the primary key column(s) - uniquely identifying a training row/case
            * the inference timestamp column - timestamp column with which to join features (temporal join timestamp)
            * context key(s) - the references to the other feature tables' primary keys (ie customer_id, location_id)
            * (optionally) the label expression - defining what the training set is trying to predict
        :param primary_keys: (List[str]) The list of columns from the training SQL that identify the training row
        :param ts_col: (Optional[str]) The timestamp column of the training SQL that identifies the inference timestamp
        :param label_col: (Optional[str]) The optional label column from the training SQL.
        :param replace: (Optional[bool]) Whether to replace an existing training set
        :param desc: (Optional[str]) An optional description of the training set
        :return:
        """

        # validate_sql()
        # register_training_context()
        pass

    def get_feature_context_keys(self, features: List[str]) -> Dict[str,List[str]]:
        """
        Returns a dictionary mapping each individual feature to its primary key(s)

        :param features: (List[str]) The list of features to get primary keys for
        :return: Dict[str, List[str]]
        """
        pass

    def get_available_features(self, training_context: Optional[str, int]) -> List[str]:
        # TODO: Webinar
        """
        Given a training context ID or name, returns the available features

        :param training_context:
        :return:
        """

    def set_feature_description(self):
        pass

    def get_feature_description(self):
        pass

    def get_training_set(self, training_context_name: str, features: List[str], start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None, return_sql: bool = False) -> Optional[SparkDF, str]:
        # TODO: Webinar
        """
        Returns the training set as a Spark Dataframe

        :param training_context_name: (str) The name of the registered training context
        :param features: (List[str]) the list of features from the feature store to be included in the training
            * NOTE: This function will error if the context SQL is missing a context key required to retrieve the\
             desired features
        :param start_time: (Optional[datetime]) The start time of the query (how far back in the data to start). Default None
            * NOTE: If start_time is None, query will start from beginning of history
        :param end_time: (Optional[datetime]) The end time of the query (how far recent in the data to get). Default None
            * NOTE: If end_time is None, query will get most recently available data
        :param return_sql: (Optional[bool]) Return the SQL statement (str) instead of the Spark DF. Defaults False
        :return: Optional[SparkDF, str]
        """
        pass

    def list_training_sets(self) -> Dict[str, Optional[str]]:
        """
        Returns a dictionary a training sets available, with the map name -> description. If there is no description,
        the value will be an emtpy string

        :return: Dict[str, Optional[str]]
        """
        pass

