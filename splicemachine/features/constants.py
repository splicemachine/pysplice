class FeatureTypes:
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
            FeatureTypes.categorical, FeatureTypes.ordinal, FeatureTypes.continuous,
        )

class SQL:
    feature_set_table = 'CREATE TABLE {schema}.{table} ({pk_columns}, {ts_columns}, {feature_columns}, ' \
                            '\nPRIMARY KEY ({pk_list}))'

    feature_set_trigger = '''
    CREATE TRIGGER {schema}.{table}_history_update 
    AFTER UPDATE ON {schema}.{table}
    REFERENCING OLD AS OLDW NEW AS NEWW
    FOR EACH ROW 
        INSERT INTO {schema}.{table}_history (ASOF_TS, UNTIL_TS, {pk_list}, {feature_list}) 
        VALUES( OLDW.LAST_UPDATE_TS, NEWW.LAST_UPDATE_TS, {old_pk_cols}, {old_feature_cols} )
    '''

    feature_set_metadata = """
    INSERT INTO FeatureStore.FeatureSet ( SchemaName, TableName, Description) VALUES ('{schema}', '{table}', '{desc}')
    """

    get_feature_set_id = "SELECT FeatureSetID FROM FeatureStore.FeatureSet " \
                         "WHERE SchemaName='{schema}' and TableName='{table}'"

    feature_set_pk_metadata = """
    INSERT INTO FeatureStore.FeatureSetKey( FeatureSetID, KeyColumnName, KeyColumnDataType) 
    VALUES ({feature_set_id}, '{pk_col_name}', '{pk_col_type}')
    """

    feature_metadata = """
    INSERT INTO FeatureStore.Feature (FeatureSetID, Name, Description, FeatureDataType, FeatureType, Tags) 
    VALUES 
    ({feature_set_id}, '{name}', '{desc}', '{feature_data_type}', '{feature_type}', '{tags}') 
    """

    get_features_in_feature_set = """
    select FeatureID,FeatureSetID,Name,Description,FeatureDataType, FeatureType,Cardinality,Tags,ComplianceLevel, 
    LastUpdateTS,LastUpdateUserID from featurestore.feature where featuresetid={featuresetid}
    """

    get_feature_sets = """
        SELECT fset.FeatureSetID, TableName, SchemaName, Description, pkcolumns, pktypes FROM FeatureStore.FeatureSet fset
        INNER JOIN 
            (
                SELECT FeatureSetID, STRING_AGG(KeyColumnName,'|') PKColumns, STRING_AGG(KeyColumnDataType,'|'
            ) pktypes 
        FROM FeatureStore.FeatureSetKey GROUP BY 1) p 
        ON fset.FeatureSetID=p.FeatureSetID 
        """
    get_training_contexts = """
    SELECT tc.ContextID, tc.Name, tc.Description, CAST(SQLText AS VARCHAR(1000)) context_sql, 
       p.PKColumns, 
       TSColumn, LabelColumn,
       c.ContextColumns               
    FROM FeatureStore.TrainingContext tc 
       INNER JOIN 
        (SELECT ContextID, STRING_AGG(KeyColumnName,',') PKColumns FROM FeatureStore.TrainingContextKey WHERE KeyType='P' GROUP BY 1)  p ON tc.ContextID=p.ContextID 
       INNER JOIN 
        (SELECT ContextID, STRING_AGG(KeyColumnName,',') ContextColumns FROM FeatureStore.TrainingContextKey WHERE KeyType='C' GROUP BY 1)  c ON tc.ContextID=c.ContextID
    """

    get_all_features = "SELECT NAME FROM FeatureStore.feature WHERE Name='{name}'"

    get_available_features = """
    SELECT f.FEATUREID, f.FEATURESETID, f.NAME, f.DESCRIPTION, f.FEATUREDATATYPE, f.FEATURETYPE, f.CARDINALITY, f.TAGS, f.COMPLIANCELEVEL, f.LASTUPDATETS, f.LASTUPDATEUSERID
          FROM FeatureStore.Feature f
          WHERE FeatureID IN
          (
              SELECT FeatureID 
              FROM
              (
                  SELECT FeatureID FROM
                    (
                        SELECT f.FeatureID, fsk.KeyCount, count(distinct fsk.KeyColumnName) ContextKeyMatchCount 
                        FROM
                            FeatureStore.TrainingContext tc 
                            INNER JOIN 
                            FeatureStore.TrainingContextKey c ON c.ContextID=tc.ContextID AND c.KeyType='C'
                            INNER JOIN 
                            ( 
                                SELECT FeatureSetId, KeyColumnName, count(*) OVER (PARTITION BY FeatureSetId) KeyCount 
                                FROM FeatureStore.FeatureSetKey 
                            )fsk ON c.KeyColumnName=fsk.KeyColumnName
                            INNER JOIN
                            FeatureStore.Feature f USING (FeatureSetID)
                        WHERE {where}
                        GROUP BY 1,2
                    )match_keys
                    WHERE ContextKeyMatchCount = KeyCount 
              )fl
          )
    """

    training_context = """
    INSERT INTO FeatureStore.TrainingContext (Name, Description, SQLText, TSColumn, LabelColumn) 
    VALUES ('{name}', '{desc}', '{sql_text}', '{ts_col}', {label_col})
    """

    get_training_context_id = """
    SELECT ContextID from FeatureStore.TrainingContext where Name='{name}'
    """

    training_context_keys = """
    INSERT INTO FeatureStore.TrainingContextKey (ContextID, KeyColumnName, KeyType)
    VALUES ({context_id}, '{key_column}', '{key_type}' )
    """

class Columns:
    feature = ['featureid', 'featuresetid', 'name', 'description', 'featuredatatype', 'featuretype',
               'cardinality', 'tags', 'compliancelevel', 'lastupdatets', 'lastupdateuserid']
    training_context = ['contextid','name','description','context_sql','pkcolumns','tscolumn','labelcolumn','contextcolumns']
    feature_set = ['featuresetid', 'tablename', 'schemaname', 'description', 'pkcolumns', 'pktypes']
    history_table_pk = ['ASOF_TS','UNTIL_TS']

