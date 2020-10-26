from enum import Enum
from splicemachine.features.constants import FeatureTypes

class Feature:
    def __init__(self, *, name, description, featuredatatype, featuretype, tags, featuresetid=None, feature_id=None, **kwargs):
        self.name = name
        self.description = description
        self.feature_data_type = featuredatatype
        self.feature_type = featuretype
        self.feature_set_id = featuresetid
        self.tags = tags
        args = {k.lower(): kwargs[k] for k in kwargs}
        self.__dict__.update(args)

    def is_categorical(self):
        return self.feature_type == FeatureTypes.categorical

    def is_continuous(self):
        return self.feature_type == FeatureTypes.continuous

    def is_ordinal(self):
        return self.feature_type == FeatureTypes.ordinal

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
