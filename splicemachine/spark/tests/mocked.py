from collections import namedtuple
import copy

schema_entry = namedtuple('schema_entry', ['name', 'dataType'])

class MockedSparkSession(object):
    """
    Mocked Spark Session object for PySpliceCtx
    """
    def __init__(self):
        self._wrapped = object()


class MockedDataFrame(object):
    """
    Mocked Dataframe
    """
    def __init__(self):
        
        self._jdf = ''
        self.columns = ['a', 'b', 'c', 'd']
        self.schema = [schema_entry('a', "IntegerType"), schema_entry('b', 'StringType'),
                       schema_entry('c', 'ByteType'), schema_entry('d', 'BooleanType')]
        self.df_schema = [schema_entry('A', 'INTEGER'), schema_entry('B', 'VARCHAR(150)'),
                          schema_entry("C", 'TINYINT'), schema_entry('D', 'BOOLEAN')]

    def withColumnRenamed(self, old_column_name, new_column_name):
        """
        Mocked rename column method
        """
        self.columns[self.columns.index(old_column_name)] = new_column_name
        return self


class MockedScalaContext(object):
    """
    This class is a Mocked Representation of the Scala SpliceMachineContext API for unit testing
    """

    def __init__(self, JDBC_URL):
        print("Class Initialized")
    
    @staticmethod
    def _generateOperationsTable(**kwargs):
        """
        Usage: _generateOperationsTable(event='get connection')
        --> {'event': 'get connection'}
        """
        return kwargs
    
    def getConnection(self):
        return self._generateOperationsTable(event='get connection')

    def tableExists(self, schemaTableName):
        schemaName, tableName = schemaTableName.split('.')
        return self._generateOperationsTable(event='table exists', schemaTableName=schemaTableName, schemaName=schemaName,
                                        tableName=tableName)

    def dropTable(self, schemaTableName):
        schemaName, tableName = schemaTableName.split('.')
        return self._generateOperationsTable(event='drop table', schemaTableName=schemaTableName, schemaName=schemaName,
                                        tableName=tableName)

    def df(self, query_string):
        return self._generateOperationsTable(event='df', query_string=query_string)

    def insert(self, dataFrame, schemaTableName):
        schemaName, tableName = schemaTableName.split('.')
        return self._generateOperationsTable(event='insert', schemaTableName=schemaTableName, schemaName=schemaName,
                                        tableName=tableName, dataFrame=dataFrame)

    def delete(self, dataFrame, schemaTableName):
        schemaName, tableName = schemaTableName.split('.')
        return self._generateOperationsTable(event='delete', schemaTableName=schemaTableName, schemaName=schemaName,
                                        tableName=tableName, dataFrame=dataFrame)

    def update(self, dataFrame, schemaTableName):
        schemaName, tableName = schemaTableName.split('.')
        return self._generateOperationsTable(event='update', schemaTableName=schemaTableName, schemaName=schemaName,
                                        tableName=tableName, dataFrame=dataFrame)

    def getSchema(self, schemaTableName):
        schemaName, tableName = schemaTableName.split('.')
        return self._generateOperationsTable(event='getSchema', schemaTableName=schemaTableName, schemaName=schemaName,
                                        tableName=tableName)
    
    def upsert(self, dataFrame, schemaTableName):
        schemaName, tableName = schemaTableName.split('.')
        return self._generateOperationsTable(event='upsert', schemaTableName=schemaTableName, schemaName=schemaName,
                                        tableName=tableName)

    def execute(self, query_string):
        return self._generateOperationsTable(event='execute', query_string=query_string)

    def executeUpdate(self, query_string):
        return self._generateOperationsTable(event='execute update', query_string=query_string)

    def internalDf(self, query_string):
        return self._generateOperationsTable(event='internal df', query_string=query_string)
    
    def truncateTable(self, schemaTableName):
        schemaName, tableName = schemaTableName.split('.')
        return self._generateOperationsTable(event='truncate', schemaTableName=schemaTableName, schemaName=schemaName,
                                        tableName=tableName)
    def analyzeSchema(self, schemaTableName):
        schemaName, tableName = schemaTableName.split('.')
        return self._generateOperationsTable(event='analyze schema', schemaTableName=schemaTableName, schemaName=schemaName,
                                        tableName=tableName) 
    def analyzeTable(self, schemaTableName, es, sp):
        schemaName, tableName = schemaTableName.split('.')
        return self._generateOperationsTable(event='analyze table', schemaTableName=schemaTableName, schemaName=schemaName,
                                        tableName=tableName, samplePercent=sp, estimateStatistics=es)    
    
    def export(self, dataframe, location, compression=False, replicationCount=1, fileEncoding=None,  fieldSeparator=None, quoteCharacter=None):
        return self._generateOperationsTable(event='export', location=location, compression=compression, replicationCount=replicationCount, fileEncoding=fileEncoding,
            fieldSeparator=fieldSeparator, quoteCharacter=quoteCharacter)
    
    def exportBinary(self, dataframe, location, compression, e_format):
        return self._generateOperationsTable(event='exportBinary', location=location, compression=compression, format=e_format)

def remove_df(spliceCtx):
    """
    Mock the df function in the PySpliceContext
    to return a default dataframe
    :param spliceCtx: the pysplice context
    :return: copy of ctx with df removed
    """
    copied_splice_ctx = copy.copy(spliceCtx) # copy so reference is not changed
    df = MockedDataFrame()
    df.columns = ['A', 'B', 'C', 'D']
    copied_splice_ctx.df = lambda sql: df  # return original
    return copied_splice_ctx

def remove_replaceDataframeSchema(spliceCtx):
    """
    Mock the replaceDataframeSchema function in
    the PySpliceContext to return the original dataframe
    :param spliceCtx: the PySpliceContext object
    :return: copy of pyspliceCtx (so mocking won't change
    original object) with method mocked
    """
    copied_splice_ctx = copy.copy(spliceCtx) # copy so reference is not changed
    copied_splice_ctx.replaceDataframeSchema = lambda df, st_name: df # return original
    return copied_splice_ctx

def remove_toUpper(spliceCtx):
    """
    Mock the toUpper function in the PySpliceContext
    to return the original dataframe (that method will
    be tested sepparately).
    :param spliceCtx: the PySpliceContext object
    :return: copy of pyspliceCtx (so mocking won't change
    original object) with method mocked
    """
    copied_splice_ctx = copy.copy(spliceCtx) # copy, so reference is not changed
    copied_splice_ctx.toUpper = lambda df: df # return original df
    return copied_splice_ctx 

def raise_error(sql):
    """
    raise an error to capture function call
    """
    raise AttributeError(sql)

def remove_execute(spliceCtx):
    """
    Mock the execute function in the PySpliceContext
    """
    copied_splice_ctx = copy.copy(spliceCtx) # copy, so reference is not changed
    copied_splice_ctx.execute = lambda sql: raise_error(sql)
    return copied_splice_ctx 

def remove_generateDBSchema(spliceCtx):
    """
    Mock the generateDBSchema helper function
    in spliceCtx
    """
    copied_splice_ctx = copy.copy(spliceCtx) # copy, so reference is not changed
    df = MockedDataFrame()
    copied_splice_ctx._generateDBSchema = lambda *args, **kwargs: df.df_schema  
    return copied_splice_ctx 

def _mocked_getCreateTableSchema(schema_table_name, new_schema=False):
    """
    mocked version of the getCreateTableSchema helper function
    """
    return "schema", "table"

def remove_getCreateTableSchema(spliceCtx):
    """
    Mock getCreateTableSchema
    """
    copied_splice_ctx = copy.copy(spliceCtx) # copy, so reference is not changed
    copied_splice_ctx._getCreateTableSchema = _mocked_getCreateTableSchema 
    return copied_splice_ctx 