from splicemachine.features.constants import FeatureTypes, SQL

class Feature:
    def __init__(self, *, name, description, feature_data_type, featuretype, tags, featuresetid=None, feature_id=None, **kwargs):
        self.name = name
        self.description = description
        self.feature_data_type = feature_data_type
        self.feature_type = featuretype
        self.feature_set_id = featuresetid
        self.feature_id = feature_id
        self.tags = tags
        args = {k.lower(): kwargs[k] for k in kwargs}
        self.__dict__.update(args)

    def is_categorical(self):
        return self.feature_type == FeatureTypes.categorical

    def is_continuous(self):
        return self.feature_type == FeatureTypes.continuous

    def is_ordinal(self):
        return self.feature_type == FeatureTypes.ordinal

    def _register_metadata(self, splice):
        """
        Registers the feature's existence in the feature store
        :return: None
        """
        feature_sql = SQL.feature_metadata.format(
            feature_set_id=self.feature_set_id, name=self.name, desc=self.description,
            feature_data_type=self.feature_data_type,
            feature_type=self.feature_type, tags=','.join(self.tags)
        )
        splice.execute(feature_sql)


    def __eq__(self, other):
        if isinstance(other, Feature):
            return self.name.lower() == other.name.lower()
        if isinstance(other, str):
            return self.name.lower() == other.lower()
        return False

    def __repr__(self):
        return self.__str__()
    def __str__(self):
        return f'Feature(FeatureID={self.__dict__.get("feature_id","None")}, ' \
               f'FeatureSetID={self.__dict__.get("feature_set_id","None")}, Name={self.name}, \n' \
               f'Description={self.description}, FeatureDataType={self.feature_data_type}, ' \
               f'FeatureType={self.feature_type}, Tags={self.tags})\n'
