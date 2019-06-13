"""
Copyright 2018 Splice Machine, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from ..context import PySpliceContext
from .mocked import MockedDataFrame, MockedSparkSession, remove_replaceDataframeSchema, \
    remove_toUpper, remove_execute, remove_getCreateTableSchema, remove_generateDBSchema, remove_df
from .utilities import Assertions
import copy

"""
Usage:
cd /path/to/tests
pytest -vv .
"""

spliceContext = PySpliceContext('jdbc://hello', MockedSparkSession(), _unit_testing=True)

class TestContext:
    """
    Some test cases that make sure all functions exist and that inputs map to outputs.
    """
    def test_getConnection(self):
        out = spliceContext.getConnection()
        assertions = {out['event']: 'get connection'}
        Assertions.dict_assertions(assertions)

    def test_tableExists(self):
        table = 'schema1.table1'
        out = spliceContext.tableExists(table)
        Assertions.basic_df_schema_assertions(out, 'table exists', table)

    def test_dropTable(self):
        table = 'schema2.table3'
        out = spliceContext.dropTable(table)
        Assertions.basic_df_schema_assertions(out, 'drop table', table)

    def test_df(self):
        query = 'SELECT * FROM table1'
        out = spliceContext.df(query)
        Assertions.query_assertions(out, 'df', query)

    def test_insert(self):
        table = 'schema.table94'
        out = remove_replaceDataframeSchema(spliceContext).insert(
            MockedDataFrame(), table)
        Assertions.basic_df_schema_assertions(out, 'insert', table)

    def test_delete(self):
        table = 'schema4.table4'
        out = spliceContext.delete(MockedDataFrame(), table)
        Assertions.basic_df_schema_assertions(out, 'delete', table)

    def test_update(self):
        table = 'schema0.table390'
        out = remove_replaceDataframeSchema(spliceContext).update(
            MockedDataFrame(), table)
        Assertions.basic_df_schema_assertions(out, 'update', table)

    def test_getSchema(self):
        table = 'schema41.table12'
        out = spliceContext.getSchema(table)
        Assertions.basic_df_schema_assertions(out, 'getSchema', table)
        
    def test_upsert(self):
        table = 'schema1.table30'
        out = remove_replaceDataframeSchema(spliceContext).upsert(
            MockedDataFrame(), table)
        Assertions.basic_df_schema_assertions(out, 'upsert', table)

    def test_execute(self):
        query = 'SELECT * FROM table2'
        out = spliceContext.execute(query)
        Assertions.query_assertions(out, 'execute', query)
    
    def test_executeUpdate(self):
        query = 'SELECT * FROM table3'
        out = spliceContext.executeUpdate(query)
        Assertions.query_assertions(out, 'execute update', query)
    
    
    def test_internalDf(self):
        query = 'SELECT * FROM table4'
        out = spliceContext.internalDf(query)
        Assertions.query_assertions(out, 'internal df', query)
    
    def test_truncateTable(self):
        table = 'schema44.table43'
        out = spliceContext.truncateTable(table)
        Assertions.basic_df_schema_assertions(out, 'truncate', table)
    
    def test_analyzeSchema(self):
        table = 'schema45.table45'
        out = spliceContext.analyzeSchema(table)
        Assertions.basic_df_schema_assertions(out, 'analyze schema', table)
    
    def test_analyzeTable(self):
        table = 'schema43.table55'
        estimateStatistics = True
        samplePercent = 0.3
        out = spliceContext.analyzeTable(table, estimateStatistics=estimateStatistics, 
            samplePercent=samplePercent)
        Assertions.basic_df_schema_assertions(out, 'analyze table', table)
        assertions = {
            out['samplePercent']: samplePercent,
            out['estimateStatistics']: estimateStatistics
        }
        Assertions.dict_assertions(assertions)

    def test_export(self):
        df = MockedDataFrame()
        location = 's3a://mysupersecretlocation'
        compression = True
        replicationCount = 5
        fileEncoding = None
        fieldSeparator = ','
        quoteCharacter = '"'
        out = spliceContext.export(df, location, compression=compression, replicationCount=replicationCount,
        fileEncoding=fileEncoding, fieldSeparator=fieldSeparator, quoteCharacter=quoteCharacter)
        assertions = {
            out['location']: location,
            out['compression']: compression,
            out['replicationCount']: replicationCount,
            out['fieldSeparator']: fieldSeparator,
            out['fileEncoding']: fileEncoding,
            out['quoteCharacter']: quoteCharacter
        }
        Assertions.dict_assertions(assertions)

    def test_exportBinary(self):
        df = MockedDataFrame()
        location = 's3a://mysuperlocation2'
        compression=True
        e_format='parquet'
        out = spliceContext.exportBinary(df, location, compression, e_format)
        assertions = {
            out['location']: location,
            out['compression']: compression,
            out['format']: e_format
        }
        Assertions.dict_assertions(assertions)

    def test_generate_db_schema_no_types(self):
        df = MockedDataFrame()
        out = spliceContext._generateDBSchema(df)
        assert out == [('A', 'INTEGER'), ('B', 'VARCHAR(150)'), 
            ('C', 'TINYINT'), ('D', 'BOOLEAN')]
    
    def test_generate_db_schema_types(self):
        df = MockedDataFrame()
        types = {
            'a': 'DOUBLE',
            'b': 'BLOB'
        }
        out = spliceContext._generateDBSchema(df, types=types)
        assert out == [('A', 'DOUBLE'), ('B', 'BLOB'), 
                        ('C', 'TINYINT'), ('D', 'BOOLEAN')]

    def test_toUpper(self):
        df = MockedDataFrame()
        out = spliceContext.toUpper(df)
        assert df.columns == ['A', 'B', 'C', 'D']
    
    def test_replaceDataframeSchema(self):
        df = MockedDataFrame()
        table = 'schema23.table954'
        out = remove_df(spliceContext).replaceDataframeSchema(df, table)
        assert out.columns == ['A', 'B', 'C', 'D']
    
    def test_getCreateTableSchema_schema_specified(self):
        df = MockedDataFrame()
        table = 'schema33.table'
        out = spliceContext._getCreateTableSchema(table)
        assert out[0] == 'SCHEMA33'
        assert out[1] == 'TABLE'
    
    def test_getCreateTableSchema_schema_notspecified(self):
        df = MockedDataFrame()
        table = 'table3'
        out = spliceContext._getCreateTableSchema(table)
        assert out[0] == 'SPLICE'
        assert out[1] == 'TABLE3'

    def test_getCreateTableSchema_create(self):
        df = MockedDataFrame()
        table = 'schema49.table329'
        try:
            out = remove_execute(spliceContext)._getCreateTableSchema(table, new_schema=True)
            raise AssertionError("Execute was not called for new schema")
        except AttributeError as e:
            assert str(e) == 'CREATE SCHEMA SCHEMA49'

    def test_dropTableIfExists(self):
        schema = 'schema333'
        table = 'table499'
        try:
            out = remove_execute(spliceContext)._dropTableIfExists(schema, table)
            raise AssertionError("Execute was not called to drop table")
        except AttributeError as e:
            assert str(e) == 'DROP TABLE IF EXISTS schema333.table499'

    def test_createTable(self):
        df = MockedDataFrame()
        try:
            out = remove_execute(remove_generateDBSchema(remove_getCreateTableSchema(
                spliceContext
            ))).createTable(df, 'schema.table')
            raise AssertionError("execute was not called to create table")
        except AttributeError as e:
            assert str(e) == """CREATE TABLE schema.table(
A INTEGER,
B VARCHAR(150),
C TINYINT,
D BOOLEAN)"""
    