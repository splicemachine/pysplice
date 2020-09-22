from enum import Enum
from splicemachine.features.constants import FeatureTypes

class Feature:
    def __init__(self, **kwargs):
        args = {k.lower(): kwargs[k] for k in kwargs}
        self.__dict__.update(args)

    def is_categorical(self):
        return self.featuretype == FeatureTypes.categorical

    def is_continuous(self):
        return self.featuretype == FeatureTypes.continuous

    def is_ordinal(self):
        return self.featuretype == FeatureTypes.ordinal

    def __str__(self):
        return f'FeatureSet(FeatureID={self.featureid}, FeatureSetID={self.featuresetid}, Name={self.name}, \n' \
               f'Description={self.description}, FeatureDataType={self.featuredatatype}, FeatureType={self.featuretype}, \n' \
               f'Cardinality={self.cardinality}, Tags={self.tags})'
