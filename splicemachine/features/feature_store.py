from typing import List, Dict, Optional

class FeatureStore:
    def __init__(self, splice):
        self.splice_ctx = splice

    def get_feature_tables(self) -> List[str]:
        """
        Returns a list of available feature tables

        :return: list[str]
        """
        pass

    def create_feature_table(self):
        """
        Creates a new feature table
        :return:
        """
        pass

    def get_training_set(self, sql: str, features: List[str], primary_keys: List[str], label_col: Optional[str] = None):
        """
        Generates a training set

        :param sql: (str) a SELECT statement that includes:
            * the primary key column(s) - uniquely identifying a training row/case
            * the

        :return: Spark DF
        """
        pass

    def register_feature(self, feature_table):
        """
        Registers a new feature as a part of a feature table
        :param feature_table:
        :return:
        """
        pass

