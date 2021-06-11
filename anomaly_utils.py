import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import datetime
from time import time
from tslearn.metrics import dtw
from tslearn.clustering import TimeSeriesKMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import stats
import numpy.ma as ma
import json
from splicemachine.notebook import *
from splicemachine.spark import ExtPySpliceContext, PySpliceContext
from datetime import datetime
from typing import *
import math
from splicemachine.mlflow_support import *
from tqdm.notebook import tqdm


## ----------------------------------- Lena's Stats functions --------------------------------------- ##

def piecewise_stat_adf_test(full_df_run_res, pw_adf_th, periods_of_consec_runs):
    nonstat_cols = []
    stat_cols = []
    for c in full_df_run_res.columns:
        p_value_col=0
        len_index = 0
        #print('For column  %s these are results' % c)
        for i in range(len(periods_of_consec_runs)-1):
            t1 = periods_of_consec_runs[i]
            t2 = periods_of_consec_runs[i+1]
            index_cond = (full_df_run_res.index >= t1) & (full_df_run_res.index < t2)
            if np.sum(index_cond) >= len_of_consecutive_period:
                x_small =full_df_run_res.loc[index_cond,c].values
                result = adfuller(x_small, autolag=None)
                if not math.isnan(result[1]):
                    p_value_col += result[1]*np.sum(index_cond)
                    len_index += np.sum(index_cond)
        if len_index>0:
            p_value_weighted = p_value_col/len_index
        else:
            p_value_weighted = 1
        #print('HERE IS THE RESULTING P_VALUE:', p_value_weighted)
        if p_value_weighted  < pw_adf_th:
            #print('Columns is stationary')
            stat_cols.append(c)
        else:
            #print('Columns is nonstationary')
            nonstat_cols.append(c)
    return nonstat_cols,stat_cols

def bruteforce_clustering(df):
    an_array = np.empty((df.shape[1],df.shape[1]))
    an_array[:] = np.NaN
    dtw_score = pd.DataFrame(an_array)
    for i in range(df.shape[1]):
        for j in range(i+1,df.shape[1]):
            dtw_score.iloc[j,i] = dtw(df.iloc[:,i], df.iloc[:,j])
    return dtw_score

def find_groups_in_dtw(dtw_score, str_columns, threshold):
    dict_dtw = {}
    my_range = dtw_score.columns
    k=0
    i=0
    work = True
    while work:
        col = my_range[i]

        collecting_ind = []
        small_dtw = dtw_score[dtw_score[col]<=threshold]
        temp = small_dtw.index.to_list()
        if len(temp)>0:
            collecting_ind.extend(temp)
            for t in temp:
                small_dtw_t = dtw_score[dtw_score[t]<=threshold]
                temp_t = small_dtw_t.index.to_list()
                if len(temp_t)>0:
                    collecting_ind.extend(temp_t)
        collecting_ind = list(dict.fromkeys(collecting_ind))
        my_range = [e for e in my_range if e not in collecting_ind]
        if len(collecting_ind)>0:
            dict_dtw[k] = {}
            dict_dtw[k]['numeric']=[col]
            dict_dtw[k]['numeric'].extend(collecting_ind)
            dict_dtw[k]['str']=[str_columns[col]]
            dict_dtw[k]['str'].extend(str_columns[collecting_ind])
            k+=1
        i+=1
        if i>=len(my_range):
            work=False
    return dict_dtw


def merge_groups_in_dict(dict_dtw, str_columns):
    new_dict ={}
    mylen = len(dict_dtw)
    k=0
    duplicate_groups = []
    all_grouped =[]
    for i in range(mylen):
        if i not in duplicate_groups:
            for j in range(i+1, mylen):
                if len(list(set(dict_dtw[j]['numeric']) & set(dict_dtw[i]['numeric']))) > 0:
                    duplicate_groups.append(j)
                    combi = list(set(dict_dtw[j]['numeric']) | set(dict_dtw[i]['numeric']))
                    combi = list(dict.fromkeys(combi))
                    try:
                        new_dict[k]['numeric'].extend(combi)
                        new_dict[k]['str'].extend(str_columns[combi])
                    except:
                        new_dict[k] ={}
                        new_dict[k]['numeric'] = [combi[0]]
                        new_dict[k]['str'] = [str_columns[combi[0]]]
                        new_dict[k]['numeric'].extend(combi[1:])
                        new_dict[k]['str'].extend(str_columns[combi[1:]])
            try:
                l = len(new_dict[k])
            except:
                new_dict[k]={}
                new_dict[k]['numeric'] = dict_dtw[i]['numeric']
                new_dict[k]['str'] = dict_dtw[i]['str']
            k+=1
    for i in range(len(new_dict)):
        new_dict[i]['numeric'] = list(dict.fromkeys(new_dict[i]['numeric']))
        new_dict[i]['str'] = list(dict.fromkeys(new_dict[i]['str']))
        all_grouped.extend(new_dict[i]['numeric'])
    return new_dict, all_grouped



def transforming_for_dtw(full_df, value_cols):
    scaler = StandardScaler()
    small_df = pd.DataFrame(scaler.fit_transform(full_df[value_cols]), columns=value_cols)
    small_df.index = full_df.index
    small_df_res = small_df.resample('1d').mean()
    small_df_res =small_df_res.ffill()
    return small_df_res


def full_grouping_dtw_score(dtw_score, str_columns, th_first, th_second):

    dict_dtw1 = find_groups_in_dtw(dtw_score,str_columns,  th_first)
    merged_groups_first, arr_all_grouped1 = merge_groups_in_dict(dict_dtw1, str_columns)

    notgrouped = dtw_score.drop(arr_all_grouped1)
    notgrouped = notgrouped.drop(arr_all_grouped1, axis=1)

    dict_dtw2 = find_groups_in_dtw(notgrouped,str_columns, th_second)
    merged_groups_second, arr_all_grouped2 = merge_groups_in_dict(dict_dtw2, str_columns)

    notgrouped_at_all = notgrouped.drop(arr_all_grouped2)
    notgrouped_at_all = notgrouped_at_all.drop(arr_all_grouped2, axis=1)

    rest ={}
    rest[0] ={}
    rest[0]['numeric'] = notgrouped_at_all.columns.tolist()
    rest[0]['str'] = str_columns[notgrouped_at_all.columns.tolist()]

    return merged_groups_first, merged_groups_second, rest

## ----------------------------------- Ben's helper functions --------------------------------------- ##
def get_tag_data(splice: PySpliceContext, sql: str) -> pd.DataFrame:
    """
    Runs the SQL to get the Spark DF of tag data, drops the nulls, cleans the data nad returns
    :param splice: PySpliceContext
    :param sql: SQL to run
    :param run_id: Run ID to log results to
    :return: Pandas DF and some stats
    """
    # Get data
    df = splice.df(sql)
    before_drop = df.count()
    print(f'Number of rows: {before_drop}')
    df = df.na.drop().orderBy('START_TS').withColumnRenamed('START_TS', 'TIME')
    # To pandas
    print('Converting to Pandas DF')
    t0 = time()
    full_df = df.toPandas().reset_index()
    print(f'Done. Took {time() - t0} seconds')
    # Get results
    after_drop = len(full_df)
    print(f'After drop: {after_drop}')
    tags_left = len(full_df.columns)
    print(f'Number of tags left: {tags_left}')

    # Convert to datetime and make index
    print('Converting time column to datetime and creating index')
    full_df['TIME'] = pd.to_datetime(full_df['TIME'])
    full_df = full_df.set_index('TIME')
    full_df = full_df.drop('index', axis=1)
    print('Done')
    print('Logging results')
    mlflow.lp('before_drop', before_drop)
    mlflow.lp('after_drop', after_drop)
    mlflow.lp('tags_left', tags_left)

    return full_df

def get_stat_nonstat_tags(full_df, full_df_10min, value_cols, pw_adf_th, periods_of_consec_runs, th_first, th_second):
    print('Getting stationary and nonstationary data')
    t0 = time()
    nonstat, stat = piecewise_stat_adf_test(full_df_10min[value_cols], pw_adf_th, periods_of_consec_runs)
    print(f'Took {time()-t0} seconds')

    if nonstat:
        print('Running on nonstat data...', end='')
        t0 = time()
        full_df_nonstat = transforming_for_dtw(full_df, nonstat)
        dtw_score_nonstat = bruteforce_clustering(full_df_nonstat)
        mg1_nonstat, mg2_nonstat, notgr_nonstat = full_grouping_dtw_score(dtw_score_nonstat,full_df_nonstat.columns,  th_first, th_second)
        print('Done.')
        print(f'Took {time()-t0} seconds')
    else:
        mg1_nonstat = mg2_nonstat = notgr_nonstat = {}

    if stat:
        print('Running on stat data...', end='')
        t0 = time()
        full_df_stat = transforming_for_dtw(full_df, stat)
        dtw_score_stat = bruteforce_clustering(full_df_stat)
        mg1_stat, mg2_stat, notgr_stat =  full_grouping_dtw_score(dtw_score_stat,full_df_stat.columns, th_first, th_second)
        print('Done.')
        print(f'Took {time()-t0} seconds')
    else:
        mg1_stat = mg2_stat = notgr_stat = {}
    return mg1_nonstat, mg2_nonstat, notgr_nonstat, mg1_stat, mg2_stat, notgr_stat


def ten_min_resample(full_df, value_cols):
    # Resample to 10 minutes
    full_df_10min = full_df[value_cols].resample('10min').mean()
    full_df_10min = full_df_10min[~full_df_10min.isna().any(1)]
    full_df_10min = full_df_10min.reset_index()
    full_df_10min['time_diff'] = (full_df_10min['TIME']- full_df_10min['TIME'].shift(1)) / np.timedelta64(1, 's')
    full_df_10min=full_df_10min.set_index('TIME')


    periods_of_consec_runs = full_df_10min[(full_df_10min['time_diff'].diff() != 0) & (full_df_10min['time_diff'] != 600)].index.tolist()
    periods_of_consec_runs.append(full_df_10min.index.max())
    full_df_10min = full_df_10min.drop('time_diff', axis=1)
    return full_df_10min, periods_of_consec_runs

def get_moments(df, sk_kurt_threshold,window=60):
    df_for_if =df.copy()
    #
    df_for_it =df_for_if.resample('1min').mean()
    for c in df.columns:
        kurt = df[c].rolling(window).kurt()
        if np.sum(kurt.isna()) <=sk_kurt_threshold:
            df_for_if['mean_'+c] =df[c].rolling(window).mean()
            df_for_if['variance_'+c] =df[c].rolling(window).var()
            df_for_if['skewness_'+c] =df[c].rolling(window).skew()
            df_for_if['kurtosis_'+c] =df[c].rolling(window).kurt()
        else:
            df_for_if['mean_'+c] =df[c].rolling(window).mean()
            df_for_if['variance_'+c] =df[c].rolling(window).var()
            df_for_if['skewness_'+c] = np.nan
            df_for_if['kurtosis_'+c] = np.nan
    return df_for_if


## ----------------------------------- Sergio and Ben Anomaly SQL functions --------------------------------------- ##

def get_time_filter_sql(OCI_SCHEMA: str, pid: int, state_names: List[str],
                        start_time: datetime = None, end_time: datetime = None) -> str:
    state_names = tuple(state_names) if len(state_names) > 1 else f"('{state_names[0]}')"
    sql = f"""
    INNER JOIN (
        SELECT START_TS 
        FROM (
            SELECT 
                START_TS, min(STATE_ACTIVE) MIN_STATE_ACTIVE
            FROM
                {OCI_SCHEMA}.PROCESS_STATE
            WHERE 
                PROCESS_ID={pid}
            AND
                STATE_NAME in {state_names}
            GROUP BY 1
        ) x
        WHERE
            MIN_STATE_ACTIVE=true
    ) t
    USING (START_TS)
    WHERE 1=1
    """
    if start_time:
        sql += f" AND START_TS >= TIMESTAMP('{str(start_time)}') "
    if end_time:
        sql += f" AND START_TS <= TIMESTAMP('{str(end_time)}') "

    return sql

def get_good_tags(splice: PySpliceContext, OCI_SCHEMA: str, tags: List[str], state_names: List[str],
                  pid: int, pct_null: float = 0.10, start_time: datetime= None, end_time: datetime = None) -> List[str]:
    """
    Returns a reduced the list of tags by removing "bad" tags
        bad = tags with greater than pct_null nulls or any values with standard deviation of 0 (constants)
    :param tags: List of original tags
    :param pid: Process ID
    :param state_names: List of state names
    """
    tag_names = tuple(tags) if len(tags) > 1 else f"('{tags[0]}')"
    inner_sql = f"""
    SELECT 
        FULL_TAG_NAME, stddev_pop(TIME_WEIGHTED_VALUE) STDDEV, count(*) TOTAL_COUNT, sum(case when TIME_WEIGHTED_VALUE is null then 1 end) NULL_COUNT 
    FROM 
        {OCI_SCHEMA}.RESAMPLED_DATA_1M  
    {get_time_filter_sql(OCI_SCHEMA, pid, state_names, start_time, end_time)}
    AND FULL_TAG_NAME in {tag_names}
    GROUP BY 1
    """

    sql = f"""
    SELECT 
        FULL_TAG_NAME
    FROM
        ({inner_sql}) xx
    WHERE
        STDDEV > 0
    AND
        (NULL_COUNT * 1.0) / TOTAL_COUNT < {pct_null}
    AND 
        NULL_COUNT is not NULL AND STDDEV IS NOT NULL
    """
    print('getting good tags')
    print(sql)
    tags = splice.df(sql).collect()
    tags = [i[0] for i in tags]
    print(f'got {len(tags)} tags')
    return tags


def generate_denormalize_sql(splice, OCI_SCHEMA, tags: List[str], state_names: List[str], pid: int, pct_null: float = 0.10, start_time: datetime= None, end_time: datetime = None) -> Tuple[str, Dict[str,str]]:
    """
    1. Reduce the list of tags by removing "bad" tags - bad = tags with greater than pct_null nulls or any values with standard deviation of 0 (constants)
    2. Generate pivot SQL

    :param tags: full original list of tags
    :param state_names: The valid state names of the process
    :param pid: The process ID to filter state names on
    :param pct_null: The tolerable null percentage for each tag. This number should be between 0-1, with 0
    being no nulls allowed, and 1 being any number of nulls allowed. Default 0.1 (10%)
    :param start_time: The start time to filter tags on
    :param end_time: The end time to filter tags on
    """
    assert 0.0 <= pct_null <= 1.0, f'pct_null must be between 0 and 1 inclusive. You set {pct_null}'

    # Store the conversions
    tag_list = get_good_tags(splice, OCI_SCHEMA, tags, state_names, pid, pct_null, start_time, end_time)
    ## COMMENTED OUT FOLLOWING LINE BECAUSE TAGNAMES are case sensitive and this is causing NULLs because it cannot find the tag name.
    #tag_list = [i.upper() for i in tag_list]
    tag_list_names = tuple(tag_list) if len(tag_list) > 1 else f"('{tag_list[0]}')"
    tag_col_names = [i.replace('-','_').replace('.','_').upper() for i in tag_list] # The tag names cleaned up to be valid column names
    tag_col_conversion = dict(zip(tag_col_names, tag_list))

    # Expressions
    case_exp = "max(CASE WHEN FULL_TAG_NAME='{tag_name}' THEN TIME_WEIGHTED_VALUE END) {tag_col_name}, " \
               "max(CASE WHEN FULL_TAG_NAME='{tag_name}' THEN VALUE_STATE END) as {tag_col_name}_STATE"

    sql = f"""
    SELECT START_TS
    """
    for tag_name, tag_col in zip(tag_list, tag_col_names):
        sql += ',' + case_exp.format(tag_name=tag_name, tag_col_name=tag_col)
    sql += f""" ,(1.0 * SUM(CASE WHEN TIME_WEIGHTED_VALUE IS NULL THEN 1 END)) / count(*) as PERCENT_NAN_TAGS
    FROM 
        {OCI_SCHEMA}.RESAMPLED_DATA_1M 
        {get_time_filter_sql(OCI_SCHEMA, pid, state_names, start_time, end_time)} 
    AND
        FULL_TAG_NAME in {tag_list_names}
    
    GROUP BY 1
    """
    return sql, tag_col_conversion

def populate_grouping_table(OCI_SCHEMA, tag_conversions,  groups, pid):
    run_sql(f'DELETE FROM {OCI_SCHEMA}.ANOMALY_GROUP WHERE ANOMALY_PROCESS_ID={pid}')
    run_sql(f'DELETE FROM {OCI_SCHEMA}.ANOMALY_GROUP_TAG WHERE ANOMALY_PROCESS_ID={pid}')
    print('-'*100,'\n')
    full_weight=0
    for group in tqdm(groups):
        if 'group_weight' not in groups[group] and 'columns_str' not in groups[group]:
            print(f'Group {group} was missing group_weight or columns_str key(s). Had the following: {groups[group]}. '
                  f'Skipping group')
            continue
        print(f'Running SQL for group {group}:\n')
        weight = groups[group]['group_weight']
        sql = f'INSERT INTO {OCI_SCHEMA}.ANOMALY_GROUP(ANOMALY_PROCESS_ID, GROUP_ID, VECM_WEIGHT) --splice-properties insertMode=UPSERT\n VALUES ({pid},{group},{weight})'
        print(sql)
        run_sql(sql)


        sql = f'INSERT INTO {OCI_SCHEMA}.ANOMALY_GROUP_TAG(ANOMALY_PROCESS_ID, GROUP_ID, FULL_TAG_NAME) --splice-properties insertMode=UPSERT\n VALUES '
        sql += ','.join(f"({pid},{group},'{tag_conversions[tag]}')" for tag in groups[group]['columns_str'])
        print(sql)
        full_weight += weight
        run_sql(sql)
        print('-'*100,'\n')
    print('Full weight:', full_weight)
    if not math.isclose(full_weight, 1):
        warnings.warn('Something is wrong. The full weight (sum of weight group weight) should be 1 but was {full_weight}')
    # assert math.isclose(full_weight, 1), f'Something is wrong. The full weight (sum of weight group weight) should be 1 but was {full_weight}'
