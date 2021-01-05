from splicemachine.features import Feature
from .constants import SQL, Columns
from .utils import clean_df
from splicemachine.spark import PySpliceContext
from typing import List, Dict

FEATURE_SET_TS_COL = '\n\tLAST_UPDATE_TS TIMESTAMP'
HISTORY_SET_TS_COL = '\n\tASOF_TS TIMESTAMP,\n\tUNTIL_TS TIMESTAMP'


class FeatureSet:
    def __init__(self, *, splice_ctx: PySpliceContext, table_name, schema_name, description,
                 primary_keys: Dict[str, str], feature_set_id=None, deployed: bool = False, **kwargs):
        self.splice_ctx = splice_ctx

        self.table_name = table_name
        self.schema_name = schema_name
        self.description = description
        self.primary_keys = primary_keys
        self.feature_set_id = feature_set_id
        self.deployed = deployed

        args = {k.lower(): kwargs[k] for k in kwargs}  # Lowercase keys
        args = {k: args[k].split(',') if 'columns' in k else args[k] for k in
                args}  # Make value a list for specific pkcolumns because Splice doesn't support Arrays
        self.__dict__.update(args)
        self.pk_columns = list(primary_keys.keys())

    def get_features(self) -> List[Feature]:
        """
        Get's all of the features from this featureset as a list of splicemachine.features.Feature

        :return: List[Feature]
        """
        features = []
        if self.feature_set_id:
            features_df = self.splice_ctx.df(SQL.get_features_in_feature_set.format(feature_set_id=self.feature_set_id))
            features_df = clean_df(features_df, Columns.feature).collect()
            for f in features_df:
                f = f.asDict()
                features.append(Feature(**f))
        return features

    def is_deployed(self):
        """
        Returns whether or not this Feature Set has been deployed (the schema.table has been created in the database)
        :return: (bool) True if the Feature Set is deployed
        """
        return self.deployed

    def remove_feature(self, feature: Feature or str):
        """
        Removes a Feature from the Feature Set. This is not yet implemented

        :param feature: The feature to remove
        """
        raise NotImplementedError

    def get_pk_schema_str(self):
        return ','.join([f'\n\t{k} {self.primary_keys[k]}' for k in self.primary_keys])

    def get_pk_column_str(self, history=False):
        if history:
            return ','.join(self.pk_columns + Columns.history_table_pk)
        return ','.join(self.pk_columns)

    def get_feature_schema_str(self):
        return ','.join([f'\n\t{f.name}  {f.feature_data_type}' for f in self.get_features()])

    def get_feature_column_str(self):
        return ','.join([f.name for f in self.get_features()])

    def _register_metadata(self):
        fset_metadata = SQL.feature_set_metadata.format(schema=self.schema_name, table=self.table_name,
                                                        desc=self.description)

        self.splice_ctx.execute(fset_metadata)
        fsid = self.splice_ctx.df(SQL.get_feature_set_id.format(schema=self.schema_name,
                                                                table=self.table_name)).collect()[0][0]
        self.feature_set_id = fsid

        for pk in self.pk_columns:
            pk_sql = SQL.feature_set_pk_metadata.format(
                feature_set_id=fsid, pk_col_name=pk.upper(), pk_col_type=self.primary_keys[pk]
            )
            self.splice_ctx.execute(pk_sql)

    def __update_deployment_status(self, status: bool):
        """
        Updated the deployment status of a feature set after deployment/undeployment
        :return: None
        """
        self.splice_ctx.execute(SQL.update_fset_deployment_status.format(status=int(status),
                                                                         feature_set_id=self.feature_set_id))


    def deploy(self, verbose=False):
        """
        Deploys the current feature set. Equivalent to calling fs.deploy(schema_name, table_name)
        """
        old_pk_cols = ','.join(f'OLDW.{p}' for p in self.pk_columns)
        old_feature_cols = ','.join(f'OLDW.{f.name}' for f in self.get_features())

        feature_set_sql = SQL.feature_set_table.format(
            schema=self.schema_name, table=self.table_name, pk_columns=self.get_pk_schema_str(),
            ts_columns=FEATURE_SET_TS_COL, feature_columns=self.get_feature_schema_str(),
            pk_list=self.get_pk_column_str()
        )

        history_sql = SQL.feature_set_table.format(
            schema=self.schema_name, table=f'{self.table_name}_history', pk_columns=self.get_pk_schema_str(),
            ts_columns=HISTORY_SET_TS_COL, feature_columns=self.get_feature_schema_str(),
            pk_list=self.get_pk_column_str(history=True))

        trigger_sql = SQL.feature_set_trigger.format(
            schema=self.schema_name, table=self.table_name, pk_list=self.get_pk_column_str(),
            feature_list=self.get_feature_column_str(), old_pk_cols=old_pk_cols, old_feature_cols=old_feature_cols)

        print('Creating Feature Set...', end=' ')
        if verbose: print('\n', feature_set_sql, '\n')
        self.splice_ctx.execute(feature_set_sql)
        print('Done.')
        print('Creating Feature Set History...', end=' ')
        if verbose: print('\n', history_sql, '\n')
        self.splice_ctx.execute(history_sql)
        print('Done.')
        print('Creating Historian Trigger...', end=' ')
        if verbose: print('\n', trigger_sql, '\n')
        self.splice_ctx.execute(trigger_sql)
        print('Done.')
        print('Updating Metadata...')
        self.__update_deployment_status(True)
        print('Done.')

    def __eq__(self, other):
        if isinstance(other, FeatureSet):
            return self.table_name == other.table_name and self.schema_name == other.schema_name
        return False

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'FeatureSet(FeatureSetID={self.__dict__.get("feature_set_id", "NA")}, SchemaName={self.schema_name}, ' \
               f'TableName={self.table_name}, Description={self.description}, PKColumns={self.pk_columns}'
