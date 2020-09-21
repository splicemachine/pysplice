from enum import Enum
from splicemachine.features.constants import FeatureTypes

class FeatureType(Enum):
    'continuous'

class Feature:
    def __init__(self):
        pass

    def is_categorical(self):
        return self.feature_type == FeatureTypes.categorical

    def is_continuous(self):
        return self.feature_type == FeatureTypes.continuous

    def is_ordinal(self):
        return self.feature_type == FeatureTypes.ordinal
