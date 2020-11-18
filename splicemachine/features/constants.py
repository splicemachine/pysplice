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
    select feature_id,feature_set_id,Name,Description,feature_data_type, feature_type,Cardinality,Tags,compliance_level, 
    last_update_ts,last_update_username from {FEATURE_STORE_SCHEMA}.feature where Name in ({{feature_names}})
    """

    get_features_in_feature_set = f"""
    select feature_id,feature_set_id,Name,Description,feature_data_type, feature_type,Cardinality,Tags,compliance_level, 
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
    get_training_contexts = f"""
    SELECT tc.context_id, tc.Name, tc.Description, CAST(SQL_text AS VARCHAR(1000)) context_sql, 
       p.pk_columns, 
       ts_column, label_column,
       c.context_columns               
    FROM {FEATURE_STORE_SCHEMA}.training_context tc 
       INNER JOIN 
        (SELECT context_id, STRING_AGG(key_column_name,',') pk_columns FROM {FEATURE_STORE_SCHEMA}.training_context_key WHERE key_type='P' GROUP BY 1)  p ON tc.context_id=p.context_id 
       INNER JOIN 
        (SELECT context_id, STRING_AGG(key_column_name,',') context_columns FROM {FEATURE_STORE_SCHEMA}.training_context_key WHERE key_type='C' GROUP BY 1)  c ON tc.context_id=c.context_id
    """

    get_all_features = f"SELECT NAME FROM {FEATURE_STORE_SCHEMA}.feature WHERE Name='{{name}}'"

    get_available_features = f"""
    SELECT f.feature_id, f.feature_set_id, f.NAME, f.DESCRIPTION, f.feature_data_type, f.feature_type, f.CARDINALITY, f.TAGS, f.compliance_level, f.last_update_ts, f.last_update_username
          FROM {FEATURE_STORE_SCHEMA}.Feature f
          WHERE feature_id IN
          (
              SELECT feature_id 
              FROM
              (
                  SELECT feature_id FROM
                    (
                        SELECT f.feature_id, fsk.KeyCount, count(distinct fsk.key_column_name) ContextKeyMatchCount 
                        FROM
                            {FEATURE_STORE_SCHEMA}.training_context tc 
                            INNER JOIN 
                            {FEATURE_STORE_SCHEMA}.training_context_key c ON c.context_id=tc.context_id AND c.key_type='C'
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
                    WHERE ContextKeyMatchCount = KeyCount 
              )fl
          )
    """

    training_context = f"""
    INSERT INTO {FEATURE_STORE_SCHEMA}.training_context (Name, Description, SQL_text, ts_column, label_column) 
    VALUES ('{{name}}', '{{desc}}', '{{sql_text}}', '{{ts_col}}', {{label_col}})
    """

    get_training_context_id = f"""
    SELECT context_id from {FEATURE_STORE_SCHEMA}.Training_Context where Name='{{name}}'
    """

    get_fset_primary_keys = f"""
    select distinct key_column_name from {FEATURE_STORE_SCHEMA}.Feature_Set_Key
    """

    training_context_keys = f"""
    INSERT INTO {FEATURE_STORE_SCHEMA}.training_context_key (Context_ID, Key_Column_Name, Key_Type)
    VALUES ({{context_id}}, '{{key_column}}', '{{key_type}}' )
    """

    update_fset_deployment_status = f"""
    UPDATE {FEATURE_STORE_SCHEMA}.feature_set set deployed={{status}} where feature_set_id = {{feature_set_id}} 
    """

class Columns:
    feature = ['feature_id', 'feature_set_id', 'name', 'description', 'feature_data_type', 'feature_type',
               'cardinality', 'tags', 'compliance_level', 'last_update_ts', 'last_update_username']
    training_context = ['context_id','name','description','context_sql','pk_columns','ts_column','label_column','context_columns']
    feature_set = ['feature_set_id', 'table_name', 'schema_name', 'description', 'pk_columns', 'pk_types', 'deployed']
    history_table_pk = ['ASOF_TS','UNTIL_TS']

