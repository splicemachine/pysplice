from splicemachine.features import Feature
from .constants import Columns
from splicemachine.spark import PySpliceContext
from typing import List, Dict

class FeatureSet:
    def __init__(self, *, splice_ctx: PySpliceContext = None, table_name, schema_name, description,
                 primary_keys: Dict[str, str], feature_set_id=None, deployed: bool = False, **kwargs):
        self.splice_ctx = splice_ctx

        self.table_name = table_name.upper()
        self.schema_name = schema_name.upper()
        self.description = description
        self.primary_keys = primary_keys
        self.feature_set_id = feature_set_id
        self.deployed = deployed

        args = {k.lower(): kwargs[k] for k in kwargs}  # Lowercase keys
        args = {k: args[k].split(',') if 'columns' in k else args[k] for k in
                args}  # Make value a list for specific pkcolumns because Splice doesn't support Arrays
        self.__dict__.update(args)
        self.pk_columns = list(primary_keys.keys())

    def is_deployed(self):
        """
        Returns whether or not this Feature Set has been deployed (the schema.table has been created in the database)
        :return: (bool) True if the Feature Set is deployed
        """
        return self.deployed

    def __eq__(self, other):
        if isinstance(other, FeatureSet):
            return self.table_name.lower() == other.table_name.lower() and \
                   self.schema_name.lower() == other.schema_name.lower()
        return False

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'FeatureSet(FeatureSetID={self.__dict__.get("feature_set_id", "NA")}, SchemaName={self.schema_name}, ' \
               f'TableName={self.table_name}, Description={self.description}, PKColumns={self.pk_columns}'
