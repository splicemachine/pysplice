class SparkConstants:
    """
    Constants for use with Splice Machine
    Spark Support
    """
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
