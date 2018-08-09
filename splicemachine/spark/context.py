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
from __future__ import print_function

from py4j.java_gateway import java_import
from pyspark.sql import DataFrame


class PySpliceContext:
    """
    This class implements a SpliceMachineContext object (similar to the SparkContext object)
    """

    def __init__(self, JDBC_URL, spark_sql_context, _unit_testing=False):
        """
        :param JDBC_URL: (string) The JDBC URL Connection String for your Splice Machine Cluster
        :param spark_sql_context: (sparkContext) A SparkContext Object for executing Spark Queries
        """
        self.jdbcurl = JDBC_URL
        self._unit_testing = _unit_testing

        if not _unit_testing:  # Private Internal Argument to Override Using JVM
            self.spark_sql_context = spark_sql_context
            self.jvm = self.spark_sql_context._sc._jvm
            java_import(self.jvm, "com.splicemachine.spark.splicemachine.*")
            java_import(self.jvm,
                        "org.apache.spark.sql.execution.datasources.jdbc.{JDBCOptions, JdbcUtils}")
            java_import(self.jvm, "scala.collection.JavaConverters._")
            self.context = self.jvm.com.splicemachine.spark.splicemachine.SplicemachineContext(
                self.jdbcurl)

        else:
            from .utils import FakeJContext
            self.spark_sql_context = spark_sql_context
            self.jvm = ''
            self.context = FakeJContext(self.jdbcurl)

    def getConnection(self):
        """
        Return a connection to the database
        """
        return self.context.getConnection()

    def tableExists(self, schema_table_name):
        """
        Check whether or not a table exists

        :param schema_table_name: (string) Table Name
        """
        return self.context.tableExists(schema_table_name)

    def dropTable(self, schema_table_name):  # works
        """
        Drop a specified table.

        :param schema_table_name: (optional) (string) schemaName.tableName
        """
        return self.context.dropTable(schema_table_name)

    def df(self, sql):
        """
        Return a Spark Dataframe from the results of a Splice Machine SQL Query

        :param sql: (string) SQL Query (eg. SELECT * FROM table1 WHERE column2 > 3)
        :return: A Spark DataFrame containing the results
        """
        if self._unit_testing:
            return self.context.df(sql)
        return DataFrame(self.context.df(sql), self.spark_sql_context)

    def insert(self, dataframe, schema_table_name):
        """
        Insert a dataframe into a table (schema.table).

        :param dataframe: (DF) The dataframe you would like to insert
        :param schema_table_name: (string) The table in which you would like to insert the RDD
        """
        return self.context.insert(dataframe._jdf, schema_table_name)

    def delete(self, dataframe, schema_table_name):
        """
        Delete records in a dataframe based on joining by primary keys from the data frame.
        Be careful with column naming and case sensitivity.

        :param dataframe: (DF) The dataframe you would like to delete
        :param schema_table_name: (string) Splice Machine Table
        """
        return self.context.delete(dataframe._jdf, schema_table_name)

    def update(self, dataframe, schema_table_name):
        """
        Update data from a dataframe for a specified schema_table_name (schema.table).
        The keys are required for the update and any other columns provided will be updated
        in the rows.

        :param dataframe: (DF) The dataframe you would like to update
        :param schema_table_name: (string) Splice Machine Table
        :return:
        """
        return self.context.update(dataframe._jdf, schema_table_name)

    def getSchema(self, schema_table_name):
        """
        Return the schema via JDBC.

        :param schema_table_name: (DF) Table name
        """
        return self.context.getSchema(schema_table_name)
