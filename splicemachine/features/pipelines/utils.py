class FeatureAgg:
    """
    A class defining the valid Aggregation functions available to features in order to create FeatureAggregations
    """
    SUM = 'sum'
    COUNT = 'count'
    AVG = 'avg'
    MEAN = 'avg'
    MIN = 'min'
    MAX = 'max'

    @staticmethod
    def get_valid():
        return (FeatureAgg.SUM, FeatureAgg.COUNT, FeatureAgg.AVG, FeatureAgg.MIN, FeatureAgg.MAX)

class AggWindow:
    """
    A class defining the valid window types available to aggregation functions for use in FeatureAggregations
    """
    SECOND = 's'
    MINUTE = 'm'
    HOUR = 'h'
    DAY = 'd'
    WEEK = 'w'
    MONTH = 'mn'
    QUARTER = 'q'
    YEAR = 'y'

    @staticmethod
    def get_valid():
        return (AggWindow.SECOND, AggWindow.MINUTE, AggWindow.HOUR, AggWindow.DAY,
                AggWindow.WEEK, AggWindow.MONTH, AggWindow.QUARTER, AggWindow.YEAR)

    @staticmethod
    def get_window(length: int, window: str):
        """
        A function to help in creating a valid window aggregation. Validates and returns the proper window aggregation
        syntax for use in FeatureAggregations

        :param length: The length of time for the window
        :param window: The window type as defined in WindowAgg.get_valid()
        :return: (str) the proper window aggregation syntax

        :Example:
            .. code-block:: python


                WindowAgg.get_window(5, WindowAgg.SECOND) -> '5s'
                WindowAgg.get_window(10, WindowAgg.MONTH) -> '10mn'
        """
        assert window in AggWindow.get_valid(), f'The provided window {window} is not valid. ' \
                                                f'Use one of {AggWindow.get_valid()}'
        assert length > 0 and type(length) == int, f'Length must be a positive, nonzero integer, but got {length}'
        return f'{length}{window}'
