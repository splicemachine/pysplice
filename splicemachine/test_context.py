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
import pyspark

from .context import PySpliceContext
from .utils import fakeDf

conf = pyspark.SparkConf().setAppName('Unit Test Python SpliceContext API')
sc = pyspark.SparkContext(conf=conf)
sqlContext = pyspark.sql.SQLContext(sc)
spliceContext = PySpliceContext('', sqlContext, _unit_testing=True)


class TestContext:
    """
    Some test cases that make sure all functions exist and that inputs map to outputs.
    Not super robust, but they all pass
    """

    def test_getConnection(self):
        out = spliceContext.getConnection()
        assert out['event'] == 'get connection'

    def test_tableExists(self):
        out = spliceContext.tableExists('schema1.table1')
        assert out['event'] == 'table exists'
        assert out['schemaTableName'] == 'schema1.table1'
        assert out['schemaName'] == 'schema1'
        assert out['tableName'] == 'table1'

    def test_dropTable(self):
        out = spliceContext.dropTable('schema2.table3')
        assert out['event'] == 'drop table'
        assert out['schemaTableName'] == 'schema2.table3'
        assert out['schemaName'] == 'schema2'
        assert out['tableName'] == 'table3'

    def test_df(self):
        out = spliceContext.df('SELECT * FROM table1')
        assert out['sql'] == 'SELECT * FROM table1'
        assert out['event'] == 'df'

    def test_insert(self):
        out = spliceContext.insert(fakeDf(), 'schema.table94')
        assert out['tableName'] == 'table94'
        assert out['schemaTableName'] == 'schema.table94'
        assert out['schemaName'] == 'schema'
        assert out['event'] == 'insert'

    def test_delete(self):
        out = spliceContext.delete(fakeDf(), 'schema4.table4')
        assert out['tableName'] == 'table4'
        assert out['schemaTableName'] == 'schema4.table4'
        assert out['schemaName'] == 'schema4'
        assert out['event'] == 'delete'

    def test_update(self):
        out = spliceContext.update(fakeDf(), 'schema0.table390')
        assert out['tableName'] == 'table390'
        assert out['schemaTableName'] == 'schema0.table390'
        assert out['schemaName'] == 'schema0'
        assert out['event'] == 'update'

    def test_getSchema(self):
        out = spliceContext.getSchema('schema41.table12')
        assert out['event'] == 'getSchema'
        assert out['schemaTableName'] == 'schema41.table12'
        assert out['schemaName'] == 'schema41'
        assert out['tableName'] == 'table12'
