from splicemachine.features.constants import FeatureType
from .utils.feature_utils import sql_to_datatype

class Feature:
    def __init__(self, *, name, description, feature_data_type, feature_type, tags, attributes, feature_set_id=None, feature_id=None, **kwargs):
        self.name = name.upper()
        self.description = description
        self.feature_data_type = sql_to_datatype(feature_data_type)
        self.feature_type = feature_type
        self.feature_set_id = feature_set_id
        self.feature_id = feature_id
        self.tags = tags
        self.attributes = attributes
        args = {k.lower(): kwargs[k] for k in kwargs}
        self.__dict__.update(args)

    def is_categorical(self):
        """
        Returns if the type of this feature is categorical
        """
        return self.feature_type == FeatureType.categorical

    def is_continuous(self):
        """
        Returns if the type of this feature is continuous
        """
        return self.feature_type == FeatureType.continuous

    def is_ordinal(self):
        """
        Returns if the type of this feature is ordinal
        """
        return self.feature_type == FeatureType.ordinal


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

    def __hash__(self):
        return hash(repr(self))

    def __lt__(self, other):
        if isinstance(other, str):
            return self.name < other
        elif isinstance(other, Feature):
            return self.name < other.name
        raise TypeError(f"< not supported between instances of Feature and {type(other)}")
