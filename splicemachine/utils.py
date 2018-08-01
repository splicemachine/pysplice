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

def _generateOperationsTable(**kwargs):
    """
    Usage: _generateOperationsTable(event='get connection')
    --> {'event': 'get connection'}
    """
    return kwargs


class fakeDf(object):
    def __init__(self):
        self._jdf = ''


class FakeJContext(object):
    """
    This class is a Fake Representation of the Scala SpliceMachineContext API for unit testing
    """

    def __init__(self, JDBC_URL):
        print("Class Initialized")

    def getConnection(self):
        return _generateOperationsTable(event='get connection')

    def tableExists(self, schemaTableName):
        schemaName, tableName = schemaTableName.split('.')
        return _generateOperationsTable(event='table exists', schemaTableName=schemaTableName, schemaName=schemaName,
                                        tableName=tableName)

    def dropTable(self, schemaTableName):
        schemaName, tableName = schemaTableName.split('.')
        return _generateOperationsTable(event='drop table', schemaTableName=schemaTableName, schemaName=schemaName,
                                        tableName=tableName)

    def df(self, sql):
        return _generateOperationsTable(event='df', sql=sql)

    def insert(self, dataFrame, schemaTableName):
        schemaName, tableName = schemaTableName.split('.')
        return _generateOperationsTable(event='insert', schemaTableName=schemaTableName, schemaName=schemaName,
                                        tableName=tableName, dataFrame=dataFrame)

    def delete(self, dataFrame, schemaTableName):
        schemaName, tableName = schemaTableName.split('.')
        return _generateOperationsTable(event='delete', schemaTableName=schemaTableName, schemaName=schemaName,
                                        tableName=tableName, dataFrame=dataFrame)

    def update(self, dataFrame, schemaTableName):
        schemaName, tableName = schemaTableName.split('.')
        return _generateOperationsTable(event='update', schemaTableName=schemaTableName, schemaName=schemaName,
                                        tableName=tableName, dataFrame=dataFrame)

    def getSchema(self, schemaTableName):
        schemaName, tableName = schemaTableName.split('.')
        return _generateOperationsTable(event='getSchema', schemaTableName=schemaTableName, schemaName=schemaName,
                                        tableName=tableName)
