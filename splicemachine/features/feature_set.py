from splicemachine.features import Feature, SQL
from splicemachine.spark import PySpliceContext
from typing import List, Dict

FEATURE_SET_TS_COL = 'LAST_UPDATE_TS TIMESTAMP'
HISTORY_SET_TS_COL = 'ASOF_TS TIMESTAMP, UNTIL_TS TIMESTAMP'

class FeatureSet:
    def __init__(self, *, splice_ctx: PySpliceContext, tablename, schemaname,
                 description, primary_keys: Dict[str,str], featuresetid = None, **kwargs):
        self.splice_ctx = splice_ctx

        # FIXME: Set instance variables
        self.tablename = tablename
        self.schemaname = schemaname
        self.description = description
        self.primary_keys = primary_keys
        self.featuresetid = featuresetid

        args = {k.lower(): kwargs[k] for k in kwargs} # Lowercase keys
        args = {k: args[k].split(',') if 'columns' in k else args[k] for k in args} # Make value a list for specific pkcolumns because Splice doesn't support Arrays
        self.__dict__.update(args)
        self.pkcolumns = list(primary_keys.keys())

        self.features = []
        self.get_features()

    def get_features(self):
        features = self.splice_ctx.df(f'select FeatureID,FeatureSetID,Name,Description,FeatureDataType, FeatureType,Cardinality,Tags,ComplianceLevel, LastUpdateTS,LastUpdateUserID from featurestore.feature where featuresetid={self.featuresetid}').collect()
        for f in features:
            f = f.asDict()
            self.features.append(Feature(**f))

    def add_feature(self, feature: Feature):
        # TODO: Make sure feature name doesn't exist in feature store
        self.features.append(feature)

    def get_feature_by_name(self, feature_name: str):
        return [f for f in self.features if f.name == feature_name][0]

    def remove_feature(self, feature: Feature or str):
        if isinstance(feature, str):
            feature = self.get_feature_by_name(feature)
        self.features.remove(self.features.index(feature))



    def get_pk_schema_str(self):
        return ','.join([f'{k} {self.primary_keys[k]}' for k in self.primary_keys])

    def get_pk_column_str(self, history=False):
        if history:
            return ','.join(self.pkcolumns + ['ASOF_TS','UNTIL_TS'])
        return ','.join(self.pkcolumns)

    def get_feature_schema_str(self):
        return ','.join([f'{f.name}  {f.featuredatatype}' for f in self.features])

    def get_feature_column_str(self):
        return ','.join([f.name for f in self.features])


    def register_metadata(self):
        fset_metadata = SQL.feature_set_metadata.format(schema=self.schemaname, table=self.tablename,
                                                        desc=self.description)

        self.splice_ctx.execute(fset_metadata)
        fsid = self.splice_ctx.df(SQL.get_feature_set_id.format(schema=self.schemaname,
                                                               table=self.tablename)).collect()[0][0]

        for pk in self.pkcolumns:
            pk_sql = SQL.feature_set_pk_metadata.format(
                feature_set_id=fsid, pk_col_name=pk, pk_col_type=self.primary_keys[pk]
            )
            self.splice_ctx.execute(pk_sql)
        for f in self.features:
            feature_sql = SQL.feature_metadata.format(
                feature_set_id=fsid, name=f.name, desc=f.description, feature_data_type=f.featuredatatype,
                feature_type=f.featuretype, tags=f.tags, compliance_level=f.compliancelevel
            )
            self.splice_ctx.execute(feature_sql)


    def deploy(self):
        old_pk_cols = ','.join(f'OLDW.{p}' for p in self.pkcolumns)
        old_feature_cols = ','.join(f'OLDW.{f.name}' for f in self.features)

        feature_set_sql = SQL.feature_set_table.format(
            schema=self.schemaname, table=self.tablename, key_columns=self.get_pk_schema_str(),
            ts_columns=FEATURE_SET_TS_COL, feature_columns=self.get_feature_schema_str(),
            pk_list=self.get_pk_column_str()
        )

        history_sql = SQL.feature_set_table.format(
            schema=self.schemaname, table=f'{self.tablename}_history', key_columns=self.get_pk_schema_str(),
            ts_columns=HISTORY_SET_TS_COL,feature_columns=self.get_feature_schema_str(),
            pk_list=self.get_pk_column_str(history=True))

        trigger_sql = SQL.feature_set_trigger.format(
            schema=self.schemaname, table=self.tablename,pk_list=self.get_pk_column_str(),
            feature_list = self.get_feature_column_str(), old_pk_cols=old_pk_cols,old_feature_cols=old_feature_cols)

        print('Creating Feature Set...',end=' ')
        self.splice_ctx.execute(feature_set_sql)
        print('Done.')
        print('Creating Feature Set History...',end=' ')
        self.splice_ctx.execute(history_sql)
        print('Done.')
        print('Creating Historian Trigger...',end=' ')
        self.splice_ctx.execute(trigger_sql)
        print('Done.')




    def __repr__(self):
        return str(self.__dict__)
    def __str__(self):
        return f'FeatureSet(FeatureSetID={self.featuresetid}, SchemaName={self.schemaname}, TableName={self.tablename}, Description={self.description}, PKColumns={self.pkcolumns}, Features={[f.name for f in self.features]})'

