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
    feature_set_table = f"""
    CREATE TABLE {{schema}}.{{table}} ({{pk_columns}}, {{ts_columns}}, {{feature_columns}}, PRIMARY KEY ({{pk_list}}))
    """


    feature_set_trigger = f'''
    CREATE TRIGGER {{schema}}.{{table}}_history_update 
    AFTER UPDATE ON {{schema}}.{{table}}
    REFERENCING OLD AS OLDW NEW AS NEWW
    FOR EACH ROW 
        INSERT INTO {{schema}}.{{table}}_history (ASOF_TS, UNTIL_TS, {{pk_list}}, {{feature_list}}) 
        VALUES( OLDW.LAST_UPDATE_TS, NEWW.LAST_UPDATE_TS, {{old_pk_cols}}, {{old_feature_cols}} )
    '''

    feature_set_metadata = f"""
    INSERT INTO {FEATURE_STORE_SCHEMA}.feature_set ( schema_name, table_name, Description) VALUES ('{{schema}}', '{{table}}', '{{desc}}')
    """

    get_feature_set_id = f"""
    SELECT feature_set_id FROM {FEATURE_STORE_SCHEMA}.feature_set
    WHERE schema_name='{{schema}}' and table_name='{{table}}'
    """

    feature_set_pk_metadata = f"""
    INSERT INTO {FEATURE_STORE_SCHEMA}.feature_set_key( feature_set_id, key_column_name, key_column_data_type) 
    VALUES ({{feature_set_id}}, '{{pk_col_name}}', '{{pk_col_type}}')
    """

    feature_metadata = f"""
    INSERT INTO {FEATURE_STORE_SCHEMA}.Feature (feature_set_id, Name, Description, feature_data_type, feature_type, Tags) 
    VALUES 
    ({{feature_set_id}}, '{{name}}', '{{desc}}', '{{feature_data_type}}', '{{feature_type}}', '{{tags}}') 
    """

    get_features_by_name = f"""
    select fset.schema_name,fset.table_name,f.Name,f.Description,f.feature_data_type,f.feature_type,f.Tags,
    f.compliance_level,f.last_update_ts,f.last_update_username,f.feature_id,f.feature_set_id
    from {FEATURE_STORE_SCHEMA}.feature f
    join {FEATURE_STORE_SCHEMA}.feature_set fset on f.feature_set_id=fset.feature_set_id
    where {{where}}
    """

    get_features_in_feature_set = f"""
    select feature_id,feature_set_id,Name,Description,feature_data_type, feature_type,Tags,compliance_level, 
    last_update_ts,last_update_username from {FEATURE_STORE_SCHEMA}.feature where feature_set_id={{feature_set_id}}
    """

    get_feature_sets = f"""
        SELECT fset.feature_set_id, table_name, schema_name, Description, pk_columns, pk_types, deployed FROM {FEATURE_STORE_SCHEMA}.feature_set fset
        INNER JOIN 
            (
                SELECT feature_set_id, STRING_AGG(key_column_name,'|') pk_columns, STRING_AGG(key_column_data_type,'|'
            ) pk_types 
        FROM {FEATURE_STORE_SCHEMA}.feature_set_key GROUP BY 1) p 
        ON fset.feature_set_id=p.feature_set_id 
        """

    get_training_views = f"""
    SELECT tc.view_id, tc.Name, tc.Description, CAST(SQL_text AS VARCHAR(1000)) view_sql, 
       p.pk_columns, 
       ts_column, label_column,
       c.join_columns               
    FROM {FEATURE_STORE_SCHEMA}.training_view tc 
       INNER JOIN 
        (SELECT view_id, STRING_AGG(key_column_name,',') pk_columns FROM {FEATURE_STORE_SCHEMA}.training_view_key WHERE key_type='P' GROUP BY 1)  p ON tc.view_id=p.view_id 
       INNER JOIN 
        (SELECT view_id, STRING_AGG(key_column_name,',') join_columns FROM {FEATURE_STORE_SCHEMA}.training_view_key WHERE key_type='J' GROUP BY 1)  c ON tc.view_id=c.view_id
    """

    get_feature_set_join_keys = f"""
    SELECT fset.feature_set_id, schema_name, table_name, pk_columns FROM {FEATURE_STORE_SCHEMA}.feature_set fset
    INNER JOIN 
        (
            SELECT feature_set_id, STRING_AGG(key_column_name,'|') pk_columns, STRING_AGG(key_column_data_type,'|') pk_types 
            FROM {FEATURE_STORE_SCHEMA}.feature_set_key GROUP BY 1
        ) p 
    ON fset.feature_set_id=p.feature_set_id 
    WHERE fset.feature_set_id in (select feature_set_id from {FEATURE_STORE_SCHEMA}.feature where name in {{names}} )
    ORDER BY schema_name, table_name
    """

    get_all_features = f"SELECT NAME FROM {FEATURE_STORE_SCHEMA}.feature WHERE Name='{{name}}'"

    get_training_view_features = f"""
    SELECT f.feature_id, f.feature_set_id, f.NAME, f.DESCRIPTION, f.feature_data_type, f.feature_type, f.TAGS, f.compliance_level, f.last_update_ts, f.last_update_username
          FROM {FEATURE_STORE_SCHEMA}.Feature f
          WHERE feature_id IN
          (
              SELECT feature_id 
              FROM
              (
                  SELECT feature_id FROM
                    (
                        SELECT f.feature_id, fsk.KeyCount, count(distinct fsk.key_column_name) JoinKeyMatchCount 
                        FROM
                            {FEATURE_STORE_SCHEMA}.training_view tc 
                            INNER JOIN 
                            {FEATURE_STORE_SCHEMA}.training_view_key c ON c.view_id=tc.view_id AND c.key_type='J'
                            INNER JOIN 
                            ( 
                                SELECT feature_set_id, key_column_name, count(*) OVER (PARTITION BY feature_set_id) KeyCount 
                                FROM {FEATURE_STORE_SCHEMA}.feature_set_key 
                            )fsk ON c.key_column_name=fsk.key_column_name
                            INNER JOIN
                            {FEATURE_STORE_SCHEMA}.Feature f USING (feature_set_id)
                        WHERE {{where}}
                        GROUP BY 1,2
                    )match_keys
                    WHERE JoinKeyMatchCount = KeyCount 
              )fl
          )
    """

    training_view = f"""
    INSERT INTO {FEATURE_STORE_SCHEMA}.training_view (Name, Description, SQL_text, ts_column, label_column) 
    VALUES ('{{name}}', '{{desc}}', '{{sql_text}}', '{{ts_col}}', {{label_col}})
    """

    get_training_view_id = f"""
    SELECT view_id from {FEATURE_STORE_SCHEMA}.training_view where Name='{{name}}'
    """

    get_fset_primary_keys = f"""
    select distinct key_column_name from {FEATURE_STORE_SCHEMA}.Feature_Set_Key
    """

    training_view_keys = f"""
    INSERT INTO {FEATURE_STORE_SCHEMA}.training_view_key (View_ID, Key_Column_Name, Key_Type)
    VALUES ({{view_id}}, '{{key_column}}', '{{key_type}}' )
    """

    update_fset_deployment_status = f"""
    UPDATE {FEATURE_STORE_SCHEMA}.feature_set set deployed={{status}} where feature_set_id = {{feature_set_id}} 
    """

    training_set = f"""
    INSERT INTO {FEATURE_STORE_SCHEMA}.training_set (name, view_id ) 
    VALUES ('{{name}}', {{view_id}})
    """

    get_training_set_id = f"""
    SELECT training_set_id from {FEATURE_STORE_SCHEMA}.training_set where name='{{name}}'
    """

    training_set_feature = f"""
    INSERT INTO {FEATURE_STORE_SCHEMA}.training_set_feature (training_set_id, feature_id ) 
    VALUES 
    ({{training_set_id}}, {{feature_id}}) 
    """

    model_deployment = f"""
    INSERT INTO {FEATURE_STORE_SCHEMA}.deployment (model_schema_name, model_table_name, training_set_id, training_set_start_ts, training_set_end_ts, run_id ) 
    VALUES 
    ('{{schema_name}}','{{table_name}}',{{training_set_id}},'{{start_ts}}','{{end_ts}}', '{{run_id}}') 
    """

    training_set_feature_stats = f"""
    INSERT INTO {FEATURE_STORE_SCHEMA}.training_set_feature_stats ( training_set_id, training_set_start_ts, training_set_end_ts, feature_id, feature_cardinality, feature_histogram, feature_mean, feature_median, feature_count, feature_stddev) 
    VALUES 
    ({{training_set_id}}, {{training_set_start_ts}}, {{training_set_end_ts}}, {{feature_id}}, {{feature_cardinality}}, {{feature_histogram}}, {{feature_mean}}, {{feature_median}}, {{feature_count}}, {{feature_stddev}}) 
    """

    deployment_feature_stats = f"""
    INSERT INTO {FEATURE_STORE_SCHEMA}.deployment_feature_stats ( model_schema_name, model_table_name, model_start_ts, model_end_ts, feature_id, feature_cardinality, feature_histogram, feature_mean, feature_median, feature_count, feature_stddev) 
    VALUES 
    ({{model_schema_name}}, {{model_table_name}}, {{model_start_ts}}, {{model_end_ts}}, {{feature_id}}, {{feature_cardinality}}, {{feature_histogram}}, {{feature_mean}}, {{feature_median}}, {{feature_count}}, {{feature_stddev}}) 
    """

    get_feature_vector = """
    SELECT {feature_names} FROM {feature_sets} WHERE 
    """

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

