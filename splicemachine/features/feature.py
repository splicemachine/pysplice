from enum import Enum
from splicemachine.features.constants import FeatureTypes

class Feature:
    def __init__(self, *, name, description, featuredatatype, featuretype, tags, **kwargs):
        self.name = name
        self.description = description
        self.featuredatatype = featuredatatype
        self.featuretype = featuretype
        self.tags = tags
        args = {k.lower(): kwargs[k] for k in kwargs}
        self.__dict__.update(args)

    def is_categorical(self):
        return self.featuretype == FeatureTypes.categorical

    def is_continuous(self):
        return self.featuretype == FeatureTypes.continuous

    def is_ordinal(self):
        return self.featuretype == FeatureTypes.ordinal

    def __eq__(self, other):
        if isinstance(other, Feature):
            return self.name == other.name
        if isinstance(other, str):
            return self.name == other
        return False

    def __str__(self):
        return f'Feature(FeatureID={self.__dict__.get("featureid","None")}, ' \
               f'FeatureSetID={self.__dict__.get("featuresetid","None")}, Name={self.name}, \n' \
               f'Description={self.description}, FeatureDataType={self.featuredatatype}, ' \
               f'FeatureType={self.featuretype}, Tags={self.tags})\n'
