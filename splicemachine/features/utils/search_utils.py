from splicemachine.spark.utils import spark_df_size
import pandas as pd
from ipywidgets import widgets, Layout, interact
from IPython.display import display, clear_output
import re

import warnings

try:
    import pandas_profiling, spark_df_profiling
except:
    warnings.warn('You do not have the necessary extensions for these functions (pandas_profiling, spark_df_profiling). '
                  'Please run `pip install splicemachine[notebook]` to use these functions.')

def __filter_df(pdf, search_input) -> pd.DataFrame:
    """
    The filtering function for the search functionality. This enables users to search with | or & in the internal
    and external searches and returns the resulting rows

    :param pdf: The full dataframe
    :param search_input: The filters and operations on them
    :return: The resulting dataframe
    """
    if not search_input:
        return pdf

    res_df = pdf

    # Add an & at the beginning because the first word is exclusize
    ops = ['&'] + [i for i in search_input if i in ('&','|')] # Need to know if each one is an "and" or an "or"

    for op,word in zip(ops, re.split('\\&|\\|', search_input)):
        word = word.strip()
        try:
            temp_df = pdf[pdf['name'].str.contains(word, case=False) |
                          pdf['tags'].astype('str').str.contains(word, case=False, regex=False) |
                          pdf['attributes'].astype('str').str.contains(word, case=False, regex=False) |
                          pdf['feature_set_name'].str.contains(word, case=False, regex=False)]
            res_df = pd.concat([res_df, temp_df]) if op=='|' else res_df[res_df.name.isin(temp_df.name)]
        except: # The user used invalid regex, set result to None
            temp_df = pd.DataFrame([])
            res_df = pd.concat([res_df, temp_df])


    res_df.reset_index(drop=True, inplace=True)
    return res_df


def feature_search_internal(fs, pandas_profile=True):
    """
    The internal (Splice managed notebooks) feature search for the Splice Machine Feature Store
    """
    from beakerx import TableDisplay
    from beakerx.object import beakerx
    beakerx.pandas_display_table()

    pdf = fs.get_features_by_name()
    pdf = pdf[['name', 'feature_type', 'feature_data_type', 'description','feature_set_name','tags',
               'attributes','last_update_ts','last_update_username','compliance_level']]

    ############################################################################################
    searchText = widgets.Text(layout=Layout(width='80%'), description='Search:')

    def handle_submit(sender):
        res_df = __filter_df(pdf, searchText.value)
        table = TableDisplay(res_df)
        redisplay(table)

    searchText.on_submit(handle_submit)

    def on_feature_select( row, col, tabledisplay):
        redisplay(tabledisplay)
        feature_name = tabledisplay.values[row][0]
        print('Generating Data Report...')
        data = fs.get_training_set([feature_name], current_values_only=True).cache()
        if pandas_profile:
            df_size = spark_df_size(data)
            print('Gathering data')
            if df_size >= 5e8: # It's too big for pandas
                print("Dataset is too large. Profiling with Spark instead")
                display(spark_df_profiling.ProfileReport(data.cache(), explorative=False))
            else:
                print('Profiling Data')
                display(pandas_profiling.ProfileReport(data.toPandas(), explorative=False))
        else:
            print('Profiling Data')
            display(spark_df_profiling.ProfileReport(data, explorative=True))

    def redisplay(td):
        clear_output(wait=True)
        print('Filter on Feature name, tags, attributes, or feature set name. Search multiple values '
              'with "&" and "|" Enter a single Feature name for a detailed report. ')
        display(searchText)
        td.setDoubleClickAction(on_feature_select)
#         td.setColumnFrozen('name',True)
        display(td)


    table_data=pdf
    table = TableDisplay(table_data)
    redisplay(table)


def feature_search_external(fs, pandas_profile=True):
    """
    The external (Not Splice managed notebooks) feature search for the Splice Machine Feature Store

    :param pandas_profile: If you want to run feature level profiling with Pandas or Spark. If pandas is set to True,
        but the size of the Feature data is too large, it will fall back to spark
    """

    pdf = fs.get_features_by_name()
    pdf = pdf[['name', 'feature_type', 'feature_data_type', 'description','feature_set_name','tags',
               'attributes','last_update_ts','last_update_username','compliance_level']]

    @interact
    def column_search(Filter=''):
        print('Filter on Feature name, tags, attributes, or feature set name. Search multiple values '
              'with "&" and "|" Enter a single Feature name for a detailed report. ')

        res_df = __filter_df(pdf, Filter)

        if len(res_df) == 1:
            print("Generating Report...")
            col_name = res_df['name'].values[0]
            print(col_name)
            data = fs.get_training_set([col_name], current_values_only=True).cache()
            print('Gathering data')
            df_size = spark_df_size(data)
            print('Profiling Data')
            if pandas_profile:
                if df_size >= 5e8: # It's too big for pandas
                    print("Dataset is too large. Profiling with Spark instead")
                    display(spark_df_profiling.ProfileReport(data.cache(), explorative=True))
                else:
                    display(pandas_profiling.ProfileReport(data.toPandas(), explorative=True))
            else:
                display(spark_df_profiling.ProfileReport(data.cache(), explorative=True))
        return res_df
