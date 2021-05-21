"""
A set of utility functions for calculating drift of deployed models
"""
import datetime as datetime
import matplotlib.pyplot as plt
from datetime import datetime
import pyspark.sql.functions as f
from pyspark_dist_explore import distplot
from itertools import count, islice


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
    start_secs = (start - datetime(1970, 1, 1)).total_seconds()
    end_secs = (end - datetime(1970, 1, 1)).total_seconds()
    dates = [datetime.fromtimestamp(el) for el in
             islice(count(start_secs, (end_secs - start_secs) / number), number + 1)]
    return zip(dates, dates[1:])

def build_feature_drift_plot(features, training_set_df, model_table_df):
    """
    Displays feature by feature comparison of distributions between the training set and the model inputs.
    :param features: list of features to analyze
    :param training_set_df: the dataframe used for training the model that contains all the features to analyze
    :param model_table_df: the dataframe with the content of the model table containing all input features
    """
    final_features = [f for f in features if f in model_table_df.columns]
    # prep plot area
    n_bins = 15
    num_features = len(final_features)
    n_rows = int(num_features / 5)
    if num_features % 5 > 0:
        n_rows = n_rows + 1
    fig, axes = plt.subplots(nrows=n_rows, ncols=5, figsize=(30, 10 * n_rows))
    axes = axes.flatten()
    # calculate combined plots for each feature
    for plot, f in enumerate(final_features):
        add_feature_plot(axes[plot], training_set_df, model_table_df, f, n_bins)
    plt.show()

def build_model_drift_plot( model_table_df, time_intervals):
    """
    Displays model prediction distribution plots split into multiple time intervals.
    :param model_table_df: dataframe containing columns EVAL_TIME and PREDICTION
    :param time_intervals: number of time intervals to display
    :return:
    """
    min_ts = model_table_df.first()['EVAL_TIME']
    max_ts = model_table_df.orderBy(f.col("EVAL_TIME").desc()).first()['EVAL_TIME']
    if max_ts > min_ts:
        intervals = datetime_range_split(min_ts, max_ts, time_intervals)
        n_rows = int(time_intervals / 5)
        if time_intervals % 5 > 0:
            n_rows = n_rows + 1
        fig, axes = plt.subplots(nrows=n_rows, ncols=5, figsize=(30, 10 * n_rows))
        axes = axes.flatten()
        for i, time_int in enumerate(intervals):
            df = model_table_df.filter((f.col('EVAL_TIME') >= time_int[0]) & (f.col('EVAL_TIME') < time_int[1]))
            distplot(axes[i], [remove_outliers(df.select(f.col('PREDICTION')), 'PREDICTION')], bins=15)
            axes[i].set_title(f"{time_int[0]}")
            axes[i].legend()
    else:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        distplot(axes, [remove_outliers(model_table_df.select(f.col('PREDICTION')), 'PREDICTION')], bins=15)
        axes.set_title(f"Predictions at {min_ts}")
        axes.legend()
