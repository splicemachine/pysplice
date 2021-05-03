from splicemachine.spark.utils import spark_df_size
import pandas as pd
from ipywidgets import widgets, Layout
from IPython.display import display, clear_output
import pandas_profiling, spark_df_profiling
from time import time
import re


def feature_search_internal(pandas_profile=True):
    """
    The internal (Splice managed notebooks) feature search for the Splice Machine Feature Store
    """
    from beakerx import TableDisplay
    from beakerx.object import beakerx
    beakerx.pandas_display_table()
    print('Filter on Feature name, tags, attributes, or feature set name. Search multiple values with "&" and "|" Enter a single Feature name for a detailed report. ')
    pdf = fs.get_features_by_name()
    pdf = pdf[['name', 'feature_type', 'feature_data_type', 'description','feature_set_name','tags','attributes','last_update_ts','last_update_username','compliance_level']]

    ############################################################################################
    searchText = widgets.Text(layout=Layout(width='80%'), description='Search:')
    display(searchText)

    def handle_submit(sender):
        res_df = pdf
        # Add an & at the beginning because the first word is exclusize
        ops = ['&'] + [i for i in searchText.value if i in ('&','|')] # Need to know if each one is an "and" or an "or"
        for op,word in zip(ops, re.split('\\&|\\|',searchText.value)):
            word = word.strip()
            temp_df = pdf[pdf['name'].str.contains(word, case=False) | pdf['tags'].astype('str').str.contains(word, case=False, regex=False) | pdf['attributes'].astype('str').str.contains(word, case=False, regex=False) | pdf['feature_set_name'].str.contains(word, case=False, regex=False)]
            res_df = pd.concat([res_df, temp_df]) if op=='|' else res_df[res_df.name.isin(temp_df.name)]

        res_df.reset_index(drop=True, inplace=True)
        table = TableDisplay(res_df)
        redisplay(table)

    searchText.on_submit(handle_submit)

    def onSelectFeature( row, col, tabledisplay):
        redisplay(tabledisplay)
        feature_name = tabledisplay.values[row][0]
        print('Generating Data Report...')
        data = fs.get_training_set([feature_name]).cache()
        if pandas_profile:
            t0 = time()
            df_size = spark_df_size(data, spark.sparkContext)
            t1 = time() - t0
#             print(f'took {t1} seconds to get size')
            print('Gathering data')
            if df_size >= 5e8: # It's too big for pandas
                print("Dataset is too large. Profiling with Spark instead")
                display(spark_df_profiling.ProfileReport(data.cache(), explorative=False))
            else:
                t0 = time()
                print('Profiling Data')
                display(pandas_profiling.ProfileReport(data.toPandas(), explorative=False))
#                 t1 = time() - t0
#                 print(f'took {t1} seconds to profile')
        else:
            t0 = time()
            print('Profiling Data')
            display(spark_df_profiling.ProfileReport(data, explorative=True))
            t1 = time() - t0
#             print(f'took {t1} seconds to profile')

    def redisplay(td):
        clear_output(wait=True)
        display(searchText)
        td.setDoubleClickAction(onSelectFeature)
#         td.setColumnFrozen('name',True)
        display(td)


    table_data=pdf
    table = TableDisplay(table_data)
    redisplay(table)




def feature_search_external(pandas_profile=True):
    """
    The external (Not Splice managed notebooks) feature search for the Splice Machine Feature Store

    :param pandas_profile: If you want to run feature level profiling with Pandas or Spark. If pandas is set to True,
        but the size of the Feature is
    """
