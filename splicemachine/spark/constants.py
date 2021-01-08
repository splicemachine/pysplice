CONVERSIONS = {
    'BinaryType': 'BLOB',
    'BooleanType': 'BOOLEAN',
    'ByteType': 'TINYINT',
    'DateType': 'DATE',
    'DoubleType': 'DOUBLE',
    'DecimalType': 'DOUBLE',
    'IntegerType': 'INTEGER',
    'LongType': 'BIGINT',
    'NullType': 'VARCHAR(50)',
    'ShortType': 'SMALLINT',
    'StringType': 'VARCHAR(5000)',
    'TimestampType': 'TIMESTAMP',
    'UnknownType': 'BLOB',
    'FloatType': 'FLOAT'
}

SQL_TYPES = ['CHAR', 'LONG VARCHAR', 'VARCHAR', 'DATE', 'TIME', 'TIMESTAMP', 'BLOB', 'CLOB', 'TEXT', 'BIGINT',
             'DECIMAL', 'DOUBLE', 'DOUBLE PRECISION', 'INTEGER', 'NUMERIC', 'REAL', 'SMALLINT', 'TINYINT', 'BOOLEAN',
             'INT', 'INTEGER']

# Sql types that are compatible with SparkML modeling
SQL_MODELING_TYPES = {'CHAR', 'LONG VARCHAR', 'VARCHAR','CLOB', 'TEXT','BIGINT', 'DECIMAL', 'DOUBLE', 'INTEGER',
                      'DOUBLE PRECISION', 'INTEGER', 'NUMERIC', 'REAL', 'SMALLINT', 'TINYINT', 'BOOLEAN', 'INT'}
