from splicemachine.features import Feature, SQL, clean_df, Columns
from splicemachine.spark import PySpliceContext
from splicemachine.mlflow_support.utilities import SpliceMachineException
from typing import List, Dict

FEATURE_SET_TS_COL = '\n\tLAST_UPDATE_TS TIMESTAMP'
HISTORY_SET_TS_COL = '\n\tASOF_TS TIMESTAMP,\n\tUNTIL_TS TIMESTAMP'

class FeatureSet:
    def __init__(self, *, splice_ctx: PySpliceContext, tablename, schemaname,
                 description, primary_keys: Dict[str,str], featuresetid = None, **kwargs):
        self.splice_ctx = splice_ctx

        # FIXME: Set instance variables
        self.table_name = tablename
        self.schema_name = schemaname
        self.description = description
        self.primary_keys = primary_keys
        self.feature_set_id = featuresetid

        args = {k.lower(): kwargs[k] for k in kwargs} # Lowercase keys
        args = {k: args[k].split(',') if 'columns' in k else args[k] for k in args} # Make value a list for specific pkcolumns because Splice doesn't support Arrays
        self.__dict__.update(args)
        self.pk_columns = list(primary_keys.keys())

        self.features = []
        self.get_features()

    def get_features(self):
        if self.feature_set_id:
            features = self.splice_ctx.df(SQL.get_features_in_feature_set.format(featuresetid=self.feature_set_id))
            features = clean_df(features, Columns.feature).collect()
            for f in features:
                f = f.asDict()
                self.features.append(Feature(**f))

    def add_feature(self, *, name, description, feature_data_type, feature_type, tags: List[str]):
        f = Feature(name=name, description=description, featuredatatype=feature_data_type,
                       featuretype=feature_type, tags=tags)
        # FIXME: This only checks within this feature set. Needs to check all features
        if f in self.features:
            raise SpliceMachineException("Feature already exists. Provide a different name")
        self.features.append(f)


    def get_feature_by_name(self, feature_name: str):
        return [f for f in self.features if f.name == feature_name][0]

    def remove_feature(self, feature: Feature or str):
        if isinstance(feature, str):
            feature = self.get_feature_by_name(feature)
        self.features.remove(feature)


    def get_pk_schema_str(self):
        return ','.join([f'\n\t{k} {self.primary_keys[k]}' for k in self.primary_keys])

    def get_pk_column_str(self, history=False):
        if history:
            return ','.join(self.pk_columns + Columns.history_table_pk)
        return ','.join(self.pk_columns)

    def get_feature_schema_str(self):
        return ','.join([f'\n\t{f.name}  {f.featuredatatype}' for f in self.features])

    def get_feature_column_str(self):
        return ','.join([f.name for f in self.features])


    def __register_metadata(self):
        fset_metadata = SQL.feature_set_metadata.format(schema=self.schema_name, table=self.table_name,
                                                        desc=self.description)

        self.splice_ctx.execute(fset_metadata)
        fsid = self.splice_ctx.df(SQL.get_feature_set_id.format(schema=self.schema_name,
                                                                table=self.table_name)).collect()[0][0]
        self.feature_set_id = fsid

        for pk in self.pk_columns:
            pk_sql = SQL.feature_set_pk_metadata.format(
                feature_set_id=fsid, pk_col_name=pk, pk_col_type=self.primary_keys[pk]
            )
            self.splice_ctx.execute(pk_sql)
        for f in self.features:
            feature_sql = SQL.feature_metadata.format(
                feature_set_id=fsid, name=f.name, desc=f.description, feature_data_type=f.featuredatatype,
                feature_type=f.featuretype, tags=','.join(f.tags)
            )
            print(feature_sql)
            self.splice_ctx.execute(feature_sql)


    def deploy(self):
        old_pk_cols = ','.join(f'OLDW.{p}' for p in self.pk_columns)
        old_feature_cols = ','.join(f'OLDW.{f.name}' for f in self.features)

        feature_set_sql = SQL.feature_set_table.format(
            schema=self.schema_name, table=self.table_name, pk_columns=self.get_pk_schema_str(),
            ts_columns=FEATURE_SET_TS_COL, feature_columns=self.get_feature_schema_str(),
            pk_list=self.get_pk_column_str()
        )

        history_sql = SQL.feature_set_table.format(
            schema=self.schema_name, table=f'{self.table_name}_history', pk_columns=self.get_pk_schema_str(),
            ts_columns=HISTORY_SET_TS_COL,feature_columns=self.get_feature_schema_str(),
            pk_list=self.get_pk_column_str(history=True))

        trigger_sql = SQL.feature_set_trigger.format(
            schema=self.schema_name, table=self.table_name,pk_list=self.get_pk_column_str(),
            feature_list = self.get_feature_column_str(), old_pk_cols=old_pk_cols,old_feature_cols=old_feature_cols)

        print('Creating Feature Set...',end=' ')
        print('\n', feature_set_sql , '\n')
        self.splice_ctx.execute(feature_set_sql)
        print('Done.')
        print('Creating Feature Set History...',end=' ')
        print('\n', history_sql, '\n')
        self.splice_ctx.execute(history_sql)
        print('Done.')
        print('Creating Historian Trigger...',end=' ')
        print('\n', trigger_sql, '\n')
        self.splice_ctx.execute(trigger_sql)
        print('Done.')
        print('Registering Metadata...')
        self.__register_metadata()
        print('Done.')




    def __repr__(self):
        return self.__str__()
    def __str__(self):
        return f'FeatureSet(FeatureSetID={self.__dict__.get("feature_set_id", "None")}, SchemaName={self.schema_name}, ' \
               f'TableName={self.table_name}, Description={self.description}, PKColumns={self.pk_columns},' \
               f'{len(self.features)} Features)'

