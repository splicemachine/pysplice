from typing import List

class FeatureAggregation:
    def __init__(self, column_name: str, agg_functions: List[str], agg_windows: List[str],
                       feature_name_prefix: str = None, agg_default_value: float = None ):
        """
        A class abstraction for defining aggregations that will generate a set of features for a feature set from a Pipeline

        :param column_name: Name of the source column from a Source SQL statement
        :param agg_function: type of SQL aggregation (sum, min, max, avg, count etc). Available aggregations
        are in the splicemachine.features.pipelines.FeatureAgg class
        :param agg_windows: list of time windows over which to perform the aggregation ("1w","5d","3m")
        available windows: "s"econd, "m"inute, "d"ay, "w"eek, "mn"onth, "q"uarter, "y"ear
        :param agg_default_value: the default value in case of a null (no activity in the window)
        available windows: "s"econd, "m"ute, "d"ay, "w"eek, "mn"onth, "q"uarter, "y"ear
        setting this parameter would create 1 new feature for every agg_window+agg_function pair specified

        The name of the features created through this FeatureAggregation will be: schema_table_sourcecol_aggfunc_aggwindow

        :Example:
            .. code-block:: python

                from splicemachine.features.pipelines import AggWindow, FeatureAgg, FeatureAggregation
                FeatureAggregation('revenue', [FeatureAgg.SUM, FeatureAgg.AVG],\
                        [AggWindow.get_window(5, AggWindow.DAY), AggWindow.get_window(10, AggWindow.SECOND)], 0.0)

        would yield:
            customer_rfm_revenue_wrate_sum_1d
            customer_rfm_revenue_wrate_sum_5w
            customer_rfm_revenue_wrate_mean_1d
            customer_rfm_revenue_wrate_mean_5w
      """
        self.column_name=column_name
        self.agg_functions= agg_functions
        self.agg_windows = agg_windows
        self.agg_default_value = agg_default_value
        self.feature_name_prefix = feature_name_prefix
