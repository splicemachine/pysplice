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
                            'PRIMARY KEY ({pk_list}))'

    feature_set_trigger = '''
    CREATE TRIGGER {schema}.{table}_history_update AFTER UPDATE ON {schema}.{table}
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

class Columns:
    feature = ['featureid', 'featuresetid', 'name', 'description', 'featuredatatype', 'featuretype',
               'cardinality', 'tags', 'compliancelevel', 'lastupdatets', 'lastupdateuserid']
    history_table_pk = ['ASOF_TS','UNTIL_TS']
