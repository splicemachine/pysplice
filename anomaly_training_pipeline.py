import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from time import time
import datetime
from tslearn.metrics import dtw
from tslearn.clustering import TimeSeriesKMeans
import math
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import stats
import numpy.ma as ma
import json

from iso_forest import isolation_forest
from splicemachine.notebook import run_sql
from splicemachine.spark import ExtPySpliceContext, PySpliceContext
from splicemachine.mlflow_support import *
from typing import Union, List
from pyspark.sql import SparkSession
import h2o
from pysparkling import *
from h2o.estimators import H2OIsolationForestEstimator

from .anomaly_utils import (generate_denormalize_sql, ten_min_resample, populate_grouping_table,
                            get_tag_data, get_stat_nonstat_tags, get_moments)
from .vecm_model import VECM

print('Getting Spark and H2O contexts')
spark = SparkSession.builder.getOrCreate()
conf = H2OConf().setInternalClusterMode()
hc = H2OContext.getOrCreate(conf)
print('Done')



def _get_valid_tags(splice, OCI_SCHEMA, process_id, source_id) -> List[str]:
    """
    Gets the tags for the given source and process IDs
    :param splice: A PySpliceContext or ExtPySpliceContext
    :param OCI_SCHEMA: :param OCI_SCHEMA: The OCI schema (OCI or OCI2 most likely)
    :param process_id: The Process ID to train on
    :param source_id: The Source ID to train on
    :return: Valid tags
    """
    tags = splice.df(f'select FULL_TAG_NAME from {OCI_SCHEMA}.TAGLOOKUPWITHIDENTITY where '
                     f'PROCESS_ID={process_id} and SOURCE={source_id}').collect()
    tags = [i[0] for i in tags]
    print(f'Starting with {len(tags)} tags')
    return tags


def run_pipeline(splice: Union[PySpliceContext, ExtPySpliceContext],
                 OCI_SCHEMA: str,
                 process_id: int,
                 source_id: int,
                 experiment_name: str,
                 percent_null: float = 0.1,
                 start_time: datetime = None,
                 end_time: datetime = None,
                 len_of_consecutive_period: int = 20,
                 th_first: int = 5,
                 th_second: int = 9,
                 qnt: float = 0.999,
                 pw_adf_th: float = 0.01,
                 sk_kurt_threshold: int = 1000,
                 rolling_window_hours: int = 10,
                 if_thr: int = 0.99):
    """
    The parent function to run the entire pipeline for training the anomaly detection models.

    :param splice: A PySpliceContext or ExtPySpliceContext
    :param OCI_SCHEMA: The OCI schema (OCI or OCI2 most likely)
    :param process_id: The Process ID to train on
    :param source_id: The Source ID to train on
    :param experiment_name: The mlflow experiment name to run under
    :param percent_null:
    :param start_time:
    :param end_time:
    :param len_of_consecutive_period:
    :param th_first:
    :param th_second:
    :param qnt:
    :param pw_adf_th:
    :param sk_kurt_threshold:
    :param rolling_window_hours:
    :param if_thr:
    :return:
    """
    tags = _get_valid_tags(splice, OCI_SCHEMA, process_id, source_id)
    tags_start = len(tags)
    state_names = ['RUNNING'] # TODO: Is this correct and static?

    sql, conversion = generate_denormalize_sql(splice, OCI_SCHEMA, tags, state_names, process_id,percent_null,
                                               start_time=start_time, end_time=end_time)

    # Start mlflow experiment
    print(f"Starting experiment {experiment_name}")
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name='VECM') as run:
        print(f'Starting run VECM ({run.info.run_uuid})')
        # Use the generated SQL to create our dataframe
        # Then drop nulls and order by start time
        print('Getting data')
        with mlflow.timer('tag data time'):
            full_df = get_tag_data(splice, sql)

        # Filter only tags that we want
        value_cols = [x for x in full_df.columns if not x.endswith('STATE') and x != 'PERCENT_NAN_TAGS']
        state_cols = [x for x in full_df.columns if x.endswith('STATE') and x != 'PERCENT_NAN_TAGS']

        # Get the 10 minute resampled data
        with mlflow.timer('10 min resample time'):
            full_df_10min, periods_of_consec_runs = ten_min_resample(full_df, value_cols)

        # Get the stationary and nonstationary tags
        print('Getting stationary and nonstationary tags time')
        with mlflow.timer('Stat/Nonstat tag collection time'):
            (mg1_nonstat, mg2_nonstat, notgr_nonstat,
             mg1_stat, mg2_stat, notgr_stat) = get_stat_nonstat_tags(full_df, full_df_10min, value_cols, pw_adf_th,
                                                        periods_of_consec_runs, th_first, th_second)

        # Train the VECM model
        # only running it on value_cols, so no columns with STATE in it
        print('Training VECM Model')
        with mlflow.timer('VECM Model train time'):
            vecm_model = VECM(quantile=qnt)
            (df_anomalies, df_group_anomalies,
             df_total_vecm_anomaly_score, all_info) = vecm_model.fit(full_df[value_cols],
                                                                     (mg1_nonstat, mg2_nonstat, mg1_stat, mg2_stat,
                                                                      notgr_nonstat,notgr_stat )
                                                                     )

        cv_params = {
            'len_of_consecutive_period':len_of_consecutive_period,
            'th_first':th_first,
            'th_second':th_second,
            'qnt':qnt,
            'pw_adf_th':pw_adf_th,
            'sk_kurt_threshold':sk_kurt_threshold,
            'rolling_window_hours':rolling_window_hours,
            'if_thr':if_thr,
            'tags_start': tags_start
        }
        mlflow.log_params(cv_params)
        json.dump(all_info, open('all_info.json', 'w'), default=str)
        mlflow.log_artifact('all_info.json')
        mlflow.log_model(vecm_model, model_lib='pyfunc')
        with open('model_load.py', 'w+') as f:
            f.write('# How to load and access the underlying model\n')
            f.write('from splicemachine.mlflow_support import *\n')
            f.write(f"pyfunc_model = mlflow.load_model('{mlflow.active_run().info.run_uuid}')\n")
            f.write('model = pyfunc_model._model_impl.python_model\n')
        mlflow.log_artifact('model_load.py')

    print('Populating grouping tables')
    populate_grouping_table(OCI_SCHEMA, tag_conversions=conversion, groups=all_info, pid=process_id)

    print('Starting Isolation Forest process')
    # creating first difference for all columns
    first_diff_cols = []
    for c in value_cols:
        full_df[c+'_diff'] = full_df[c].diff()
        first_diff_cols.append(c+'_diff')
    print('Getting moments')
    t0 = time()
    moments_df = get_moments(full_df, sk_kurt_threshold,window=60*rolling_window_hours)
    df_notdiff = moments_df[value_cols].astype('float32')
    df_diff = moments_df[first_diff_cols].astype('float32')
    print(f'Done. Took {time() - t0} seconds')
    # To Spark then to h2o, this can take 5mins
    print('Converting to Spark DF... Can take a while...', end='')
    t0 = time()
    spark_gr1 = spark.createDataFrame(df_notdiff)
    spark_gr2 = spark.createDataFrame(df_diff)
    print('Done.')
    print(f'Took {time()-t0} seconds')

    print('Converting to H2O DF... Can take a while...', end='')
    t0 = time()
    hf_gr1 = hc.asH2OFrame(spark_gr1)
    hf_gr2 = hc.asH2OFrame(spark_gr2)
    print('Done.')
    print(f'Took {time()-t0} seconds')

    cv_params ={
            'sample_rate':0.1,
            'max_depth':6,
            'ntrees':100,
            'groups':'notdiff, diff',
            'len_of_consecutive_period':len_of_consecutive_period,
            'th_first':th_first,
            'th_second':th_second,
            'qnt':qnt,
            'pw_adf_th':pw_adf_th,
            'sk_kurt_threshold':sk_kurt_threshold,
            'rolling_window_hours':rolling_window_hours,
            'if_thr':if_thr
        }
    # Starting notdiff iso forest
    print('Training notdiff Iso Forest')
    with mlflow.start_run('notdiff') as run:
        IF_gr1 = isolation_forest(hf_gr1,'notdiff', if_thr, cv_params)
    # Starting diff iso forest
    print('Training diff Iso Forest')
    with mlflow.start_run(run_name='diff'):
        IF_gr2 = isolation_forest(hf_gr2,'diff', if_thr)
