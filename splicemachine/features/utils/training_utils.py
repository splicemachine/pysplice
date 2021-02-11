from splicemachine import SpliceMachineException
from typing import List
from splicemachine.features import Feature, FeatureSet
from splicemachine.features.training_view import TrainingView

"""
A set of utility functions for creating Training Set SQL 
"""

def clean_df(df, cols):
    for old, new in zip(df.columns, cols):
        df = df.withColumnRenamed(old, new)
    return df

def dict_to_lower(dict):
    """
    Converts a dictionary to all lowercase keys

    :param dict: The dictionary
    :return: The lowercased dictionary
    """
    return {i.lower(): dict[i] for i in dict}


def _get_anchor_feature_set(features: List[Feature], feature_sets: List[FeatureSet]) -> FeatureSet:
    """
    From a dataframe of feature set rows, where each row has columns feature_set_id, schema_name, table_name
    and pk_cols where pk_cols is a pipe delimited string of Primary Key column names,
    this function finds which row has the superset of all primary key columns, raising an exception if none exist

    :param fset_keys: Pandas Dataframe containing FEATURE_SET_ID, SCHEMA_NAME, TABLE_NAME, and PK_COLUMNS, which
    is a | delimited string of column names
    :return: FeatureSet
    :raise: SpliceMachineException
    """
    # Get the Feature Set with the maximum number of primary key columns as the anchor
    anchor_fset = feature_sets[0]

    for fset in feature_sets:
        if len(fset.pk_columns) > len(anchor_fset.pk_columns):
            anchor_fset = fset

    # If Features are requested that come from Feature Sets that cannot be joined to our anchor, we will raise an
    # Exception and let the user know
    bad_features = []
    all_pk_cols = set(anchor_fset.pk_columns)
    for fset in feature_sets:
        if not set(fset.pk_columns).issubset(all_pk_cols):
            bad_features += [f.name for f in features if f.feature_set_id == fset.feature_set_id]

    if bad_features:
        raise SpliceMachineException(f"The provided features do not have a common join key."
                                     f"Remove features {bad_features} from your request")

    return anchor_fset


def _generate_training_set_history_sql(tvw: TrainingView, features: List[Feature],
                                       feature_sets: List[FeatureSet], start_time=None, end_time=None) -> str:
    """
    Generates the SQL query for creating a training set from a TrainingView and a List of Features.
    This performs the coalesces necessary to aggregate Features over time in a point-in-time consistent way

    :param tvw: The TrainingView
    :param features: List[Feature] The group of Features desired to be returned
    :param feature_sets: List[FeatureSets] the group of all Feature Sets of which Features are being selected
    :return: str the SQL necessary to execute
    """
    # SELECT clause
    sql = 'SELECT '
    for pkcol in tvw.pk_columns:  # Select primary key column(s)
        sql += f'\n\tctx.{pkcol},'

    sql += f'\n\tctx.{tvw.ts_column}, '  # Select timestamp column

    # TODO: ensure these features exist and fail gracefully if not
    for feature in features:
        sql += f'\n\tCOALESCE(fset{feature.feature_set_id}.{feature.name},fset{feature.feature_set_id}h.{feature.name}) {feature.name},'  # Collect all features over time

    # Select the optional label col
    if tvw.label_column:
        sql += f'\n\tctx.{tvw.label_column}'
    else:
        sql = sql.rstrip(',')

    # FROM clause
    sql += f'\nFROM ({tvw.view_sql}) ctx '

    # JOIN clause
    for fset in feature_sets:
        # Join Feature Set
        sql += f'\nLEFT OUTER JOIN {fset.schema_name}.{fset.table_name} fset{fset.feature_set_id} \n\tON '
        for pkcol in fset.pk_columns:
            sql += f'fset{fset.feature_set_id}.{pkcol}=ctx.{pkcol} AND '
        sql += f' ctx.{tvw.ts_column} >= fset{fset.feature_set_id}.LAST_UPDATE_TS '

        # Join Feature Set History
        sql += f'\nLEFT OUTER JOIN {fset.schema_name}.{fset.table_name}_history fset{fset.feature_set_id}h \n\tON '
        for pkcol in fset.pk_columns:
            sql += f' fset{fset.feature_set_id}h.{pkcol}=ctx.{pkcol} AND '
        sql += f' ctx.{tvw.ts_column} >= fset{fset.feature_set_id}h.ASOF_TS AND ctx.{tvw.ts_column} < fset{fset.feature_set_id}h.UNTIL_TS'

    # WHERE clause on optional start and end times
    if start_time or end_time:
        sql += '\nWHERE '
        if start_time:
            sql += f"\n\tctx.{tvw.ts_column} >= '{str(start_time)}' AND"
        if end_time:
            sql += f"\n\tctx.{tvw.ts_column} <= '{str(end_time)}'"
        sql = sql.rstrip('AND')
    return sql


def _generate_training_set_sql(features: List[Feature], feature_sets: List[FeatureSet]) -> str:
    """
    Generates the SQL query for creating a training set from a List of Features (NO TrainingView).

    :param features: List[Feature] The group of Features desired to be returned
    :param feature_sets: List of Feature Sets
    :return: str the SQL necessary to execute
    """
    anchor_fset: FeatureSet = _get_anchor_feature_set(features, feature_sets)
    alias = f'fset{anchor_fset.feature_set_id}'  # We use this a lot for joins
    anchor_fset_schema = f'{anchor_fset.schema_name}.{anchor_fset.table_name} {alias} '
    remaining_fsets = [fset for fset in feature_sets if fset != anchor_fset]

    # SELECT clause
    feature_names = ','.join([f'fset{feature.feature_set_id}.{feature.name}' for feature in features])
    # Include the pk columns of the anchor feature set
    pk_cols = ','.join([f'{alias}.{pk}' for pk in anchor_fset.pk_columns])
    all_feature_columns = feature_names + ',' + pk_cols

    sql = f'SELECT {all_feature_columns} \nFROM {anchor_fset_schema}'

    # JOIN clause
    for fset in remaining_fsets:
        # Join Feature Set
        sql += f'\nLEFT OUTER JOIN {fset.schema_name}.{fset.table_name} fset{fset.feature_set_id} \n\tON '
        for ind, pkcol in enumerate(fset.pk_columns):
            if ind > 0: sql += ' AND '  # In case of multiple columns
            sql += f'fset{fset.feature_set_id}.{pkcol}={alias}.{pkcol}'
    return sql


def _create_temp_training_view(features: List[Feature], feature_sets: List[FeatureSet]) -> TrainingView:
    """
    Internal function to create a temporary Training View for training set retrieval using a Feature Set. When
    a user created

    :param fsets: List[FeatureSet]
    :param features: List[Feature]
    :return: Generated Training View
    """
    anchor_fset = _get_anchor_feature_set(features, feature_sets)
    anchor_pk_column_sql = ','.join(anchor_fset.pk_columns)
    ts_col = 'LAST_UPDATE_TS'
    schema_table_name = f'{anchor_fset.schema_name}.{anchor_fset.table_name}_history'
    view_sql = f'SELECT {anchor_pk_column_sql}, ASOF_TS as {ts_col} FROM {schema_table_name}'
    return TrainingView(pk_columns=anchor_fset.pk_columns, ts_column=ts_col, view_sql=view_sql,
                        description=None, name=None, label_column=None)

