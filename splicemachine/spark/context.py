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
import os
from py4j.java_gateway import java_import
from pyspark.sql import DataFrame


class PySpliceContext:
    """
    This class implements a SpliceMachineContext object (similar to the SparkContext object)
    """

    def __init__(self, JDBC_URL, sparkSession, _unit_testing=False):
        """
        :param JDBC_URL: (string) The JDBC URL Connection String for your Splice Machine Cluster
        :param sparkSession: (sparkContext) A SparkSession object for talking to Spark
        """
        self.jdbcurl = JDBC_URL
        self._unit_testing = _unit_testing

        if not _unit_testing:  # Private Internal Argument to Override Using JVM
            self.spark_sql_context = sparkSession._wrapped
            self.jvm = self.spark_sql_context._sc._jvm
            java_import(self.jvm, "com.splicemachine.spark.splicemachine.*")
            java_import(self.jvm,
                        "org.apache.spark.sql.execution.datasources.jdbc.{JDBCOptions, JdbcUtils}")
            java_import(self.jvm, "scala.collection.JavaConverters._")
            java_import(self.jvm, "com.splicemachine.derby.impl.*")
            self.jvm.com.splicemachine.derby.impl.SpliceSpark.setContext(self.spark_sql_context._jsc)
            self.context = self.jvm.com.splicemachine.spark.splicemachine.SplicemachineContext(
                self.jdbcurl)

        else:
            from .utils import FakeJContext
            self.spark_sql_context = sparkSession._wrapped
            self.jvm = ''
            self.context = FakeJContext(self.jdbcurl)
    def toUpper(self, dataframe):
        """
        Returns a dataframe with all uppercase column names
        :param dataframe: A dataframe with column names to convert to uppercase
        """
        for col in dataframe.columns:
            dataframe = dataframe.withColumnRenamed(col, col.upper())
        return dataframe
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
        dataframe = self.toUpper(dataframe)
        return self.context.insert(dataframe._jdf, schema_table_name)

    def upsert(self, dataframe, schema_table_name):
        """
        Upsert the data from a dataframe into a table (schema.table).

        :param dataframe: (DF) The dataframe you would like to upsert
        :param schema_table_name: (string) The table in which you would like to upsert the RDD
        """
        dataframe = self.toUpper(dataframe)
        return self.context.upsert(dataframe._jdf, schema_table_name)

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
        dataframe = self.toUpper(dataframe)
        return self.context.update(dataframe._jdf, schema_table_name)

    def getSchema(self, schema_table_name):
        """
        Return the schema via JDBC.

        :param schema_table_name: (DF) Table name
        """
        df = self.df("select * from " + schema_table_name) #should call python method self.df instead of java method df
        schm = df.schema
        return schm

    def execute(self, query_string):
        '''
        execute a query
        :param query_string: (string) SQL Query (eg. SELECT * FROM table1 WHERE column2 > 3)
        :return:
        '''
        return self.context.execute(query_string)

    def executeUpdate(self, query_string):
        '''
        execute a dml query:(update,delete,drop,etc)
        :param query_string: (string) SQL Query (eg. SELECT * FROM table1 WHERE column2 > 3)
        :return:
        '''
        return self.context.executeUpdate(query_string)

    def internalDf(self, query_string):
        '''
        SQL to Dataframe translation.  (Lazy)
        Runs the query inside Splice Machine and sends the results to the Spark Adapter app
        :param query_string: (string) SQL Query (eg. SELECT * FROM table1 WHERE column2 > 3)
        :return: pyspark dataframe contains the result of query_string
        '''
        return DataFrame(self.context.internalDf(query_string), self.spark_sql_context)

    def truncateTable(self, schema_table_name):
        """
        truncate a table
        :param schema_table_name: the full table name in the format "schema.table_name" which will be truncated
        :return:
        """
        return self.context.truncateTable(schema_table_name)

    def analyzeSchema(self, schema_name):
        """
        analyze the schema
        :param schema_name: schema name which stats info will be collected
        :return:
        """
        return self.context.analyzeSchema(schema_name)

    def analyzeTable(self, schema_table_name, estimateStatistics=False, samplePercent=0.10):
        """
        collect stats info on a table
        :param schema_table_name: full table name in the format of "schema.table"
        :param estimateStatistics:will use estimate statistics if True
        :param samplePercent:  the percentage or rows to be sampled.
        :return:
        """
        return self.context.analyzeTable(schema_table_name, estimateStatistics, samplePercent)

    def export(self, dataframe, location, compression=False, replicationCount=1, fileEncoding=None, fieldSeparator=None,
               quoteCharacter=None):
        '''
        Export a dataFrame in CSV
        :param dataframe:
        :param location: Destination directory
        :param compression: Whether to compress the output or not
        :param replicationCount:  Replication used for HDFS write
        :param fileEncoding: fileEncoding or null, defaults to UTF-8
        :param fieldSeparator: fieldSeparator or null, defaults to ','
        :param quoteCharacter: quoteCharacter or null, defaults to '"'
        :return:
        '''
        return self.context.export(dataframe._jdf, location, compression, replicationCount, fileEncoding,
                                   fieldSeparator, quoteCharacter)

    def exportBinary(self, dataframe, location,compression, format):
        '''
        Export a dataFrame in binary format
        :param dataframe:
        :param location: Destination directory
        :param compression: Whether to compress the output or not
        :param format: Binary format to be used, currently only 'parquet' is supported
        :return:
        '''
        return self.context.exportBinary(dataframe._jdf,location,compression,format)


class SpliceMLContext(PySpliceContext):
    """
    PySpliceContext for use with the cloud service.
    Although the original pysplicecontext *will work*
    on the Cloud Service (Zeppelin Notebook), this class
    does many things for ease of use.
    """
    def __init__(self, sparkSession, useH2O=False, _unit_testing=False):
        """
        Automatically find the JDBC URL and establish a connection
        to the current Splice Machine database
        :param sparkSession: the sparksession object
        :param useH2O: whether or not to
        :param _unit_testing: whether or not we are unit testing
        """
        try:
            url = os.environ['JDBC_URL']
            PySpliceContext.__init__(self, url, sparkSession, _unit_testing)
        except Exception as e:
            print(e)
            print('The SpliceMLContext is only for use on the cloud service. Please import and use the PySpliceContext instead.\nUsage:\n\tfrom splicemachine.spark.context import PySpliceContext\n\tsplice = PySpliceContext(jdbc_url, sparkSession)')
            return -1
        if useH2O:
            from pysparkling import H2OConf, H2OContext
            h2oConf = H2OConf(sparkSession)
            h2oConf.set_fail_on_unsupported_spark_param_disabled()
            self.hc = H2OContext.getOrCreate(sparkSession, h2oConf)
