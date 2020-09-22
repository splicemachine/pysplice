from splicemachine.features import Feature
class FeatureSet:
    def __init__(self, splice_ctx, **kwargs):
        self.splice_ctx = splice_ctx
        self.features = []
        args = {k.lower(): kwargs[k] for k in kwargs} # Lowercase keys
        args = {k: args[k].split(',') if 'columns' in k else args[k] for k in args} # Make value a list for specific pkcolumns because Splice doesn't support Arrays
        self.__dict__.update(args)
        self.get_features()

    def get_features(self):
        features = self.splice_ctx.df(f'select FeatureID,FeatureSetID,Name,Description,FeatureDataType, FeatureType,Cardinality,Tags,ComplianceLevel, LastUpdateTS,LastUpdateUserID from featurestore.feature where featuresetid={self.featuresetid}').collect()
        for f in features:
            f = f.asDict()
            self.features.append(Feature(**f))

    def add_feature(self):
        pass
    def remove_feature(self):
        pass

    def submit(self):
        """
        Create and run the SQL DDL for the table and trigger(s) and update the metadata
        :return:
        """
    def __repr__(self):
        return str(self.__dict__)
    def __str__(self):
        return f'FeatureSet(FeatureSetID={self.featuresetid}, SchemaName={self.schemaname}, TableName={self.tablename}, Description={self.description}, PKColumns={self.pkcolumns}, Features={[f.name for f in self.features]})'

