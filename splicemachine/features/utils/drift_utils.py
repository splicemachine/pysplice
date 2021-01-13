"""
A set of utility functions for calculating drift of deployed models
"""
import datetime as datetime


def calculate_outlier_bounds(df, column_name):
    """
    Calculates outlier bounds based on interquartile range of distribution of values in column 'column_name'
    from data set in data frame 'df'.
    :param df: data frame containing data to be analyzed
    :param column_name: column name to analyze
    :return: dictionary with keys min, max, q1 and q3 keys and corresponding values for outlier minimum, maximum
    and 25th and 75th percentile values (q1,q3)
    """
    bounds = dict(zip(["q1", "q3"], df.approxQuantile(column_name, [0.25, 0.75], 0)))
    iqr = bounds['q3'] - bounds['q1']
    bounds['min'] = bounds['q1'] - (iqr * 1.5)
    bounds['max'] = bounds['q3'] + (iqr * 1.5)
    return bounds


def remove_outliers(df, column_name):
    """
    Calculates outlier bounds no distribution of 'column_name' values and returns a filtered data frame without
    outliers in the specified column.
    :param df: data frame with data to remove outliers from
    :param column_name: name of column to remove outliers from
    :return: input data frame filtered to remove outliers
    """
    import pyspark.sql.functions as f
    bounds = calculate_outlier_bounds(df, column_name)
    return df.filter((f.col(column_name) >= bounds['min']) & (f.col(column_name) <= bounds['max']))


def add_feature_plot(ax, train_df, model_df, feature, n_bins):
    """
    Adds a distplot of the outlier free feature values from both train_df and model_df data frames which both
    contain the feature.
    :param ax: target subplot for chart
    :param train_df: training data containing feature of interest
    :param model_df: model input data also containing feature of interest
    :param feature: name of feature to display in distribution histogram
    :param n_bins: number of bins to use in histogram plot
    :return: None
    """
    from pyspark_dist_explore import distplot
    import pyspark.sql.functions as f
    distplot(ax, [remove_outliers(train_df.select(f.col(feature).alias('training')), 'training'),
                  remove_outliers(model_df.select(f.col(feature).alias('model')), 'model')], bins=n_bins)
    ax.set_title(feature)
    ax.legend()


def datetime_range_split( start: datetime, end: datetime, number: int):
    """
    Subdivides the time frame defined by 'start' and 'end' parameters into 'number' equal time frames.
    :param start: start date time
    :param end: end date time
    :param number: number of time frames to split into
    :return: list of start/end date times
    """
    from itertools import count, islice
    start_secs = (start - datetime(1970, 1, 1)).total_seconds()
    end_secs = (end - datetime(1970, 1, 1)).total_seconds()
    dates = [datetime.fromtimestamp(el) for el in
             islice(count(start_secs, (end_secs - start_secs) / number), number + 1)]
    return zip(dates, dates[1:])