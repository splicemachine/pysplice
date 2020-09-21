class FeatureTypes:
    """
    Class containing names for
    valid feature types
    """
    categorical: str = "categorical"
    ordinal: str = "ordinal"
    continuous: str = "continuous"

    @staticmethod
    def get_valid() -> tuple:
        """
        Return a tuple of the valid feature types
        :return: (tuple) valid types
        """
        return (
            FeatureTypes.categorical, FeatureTypes.ordinal, FeatureTypes.continuous,
        )
