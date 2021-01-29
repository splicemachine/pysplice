class FeatureType:
    """
    Class containing names for
    valid feature types
    """
    categorical: str = "N"
    ordinal: str = "O"
    continuous: str = "C"

    @staticmethod
    def get_valid() -> tuple:
        """
        Return a tuple of the valid feature types
        :return: (tuple) valid types
        """
        return (
            FeatureType.categorical, FeatureType.ordinal, FeatureType.continuous,
        )

class SQL:
    FEATURE_STORE_SCHEMA = 'FeatureStore'

    get_deployment_metadata = f"""
        SELECT tv.name, d.training_set_start_ts, d.training_set_end_ts,
               string_agg(f.name,',') features
        FROM featurestore.deployment d
           INNER JOIN {FEATURE_STORE_SCHEMA}.training_set ts          ON d.training_set_id=ts.training_set_id 
           INNER JOIN {FEATURE_STORE_SCHEMA}.training_set_feature tsf ON tsf.training_set_id=d.training_set_id
           LEFT OUTER JOIN {FEATURE_STORE_SCHEMA}.training_view tv    ON tv.view_id = ts.view_id
           INNER JOIN {FEATURE_STORE_SCHEMA}.feature f                ON tsf.feature_id=f.feature_id
        WHERE d.model_schema_name = '{{schema_name}}'
          AND d.model_table_name = '{{table_name}}'
        GROUP BY 1,2,3
        """

    get_model_predictions = """
            SELECT EVAL_TIME,
                   PREDICTION 
            FROM {schema_name}.{table_name} WHERE EVAL_TIME>='{start_time}' AND EVAL_TIME<'{end_time}'
            ORDER BY EVAL_TIME
            """

class Columns:
    feature = ['feature_id', 'feature_set_id', 'name', 'description', 'feature_data_type', 'feature_type',
               'tags', 'compliance_level', 'last_update_ts', 'last_update_username']
    training_view = ['view_id','name','description','view_sql','pk_columns','ts_column','label_column','join_columns']
    feature_set = ['feature_set_id', 'table_name', 'schema_name', 'description', 'pk_columns', 'pk_types', 'deployed']
    history_table_pk = ['ASOF_TS','UNTIL_TS']