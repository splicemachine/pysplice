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

    def upsert(self, dataframe, schema_table_name):
        """
        Upsert the data from a dataframe into a table (schema.table).

        :param dataframe: (DF) The dataframe you would like to upsert
        :param schema_table_name: (string) The table in which you would like to upsert the RDD
        """
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
        return self.context.update(dataframe._jdf, schema_table_name)

    def getSchema(self, schema_table_name):
        """
        Return the schema via JDBC.

        :param schema_table_name: (DF) Table name
        """
        return self.context.getSchema(schema_table_name)

    def execute(self, query_string):
        return self.context.execute(query_string)

    def executeUpdate(self, query_string):
        return self.context.executeUpdate(query_string)

    def internalDf(self, query_string):
        return DataFrame(self.context.internalDf(query_string), self.spark_sql_context)

    def truncateTable(self, schema_table_name):
        return self.context.truncate(schema_table_name)

    def analyzeSchema(self, schema_name):
        return self.context.analyzeSchema(schema_name)

    def analyzeTable(self, schema_table_name, estimateStatistics=False, samplePercent=0.10):
        return self.context.analyzeTable(schema_table_name, estimateStatistics, samplePercent)

    def export(self, dataframe, location, compression=False, replicationCount=1, fileEncoding=None, fieldSeparator=None,
               quoteCharacter=None):
        return self.context.export(dataframe._jdf, location, compression, replicationCount, fileEncoding,
                                   fieldSeparator, quoteCharacter)

    def exportBinary(self,dataframe, location,compression, format):
        return self.context.exportBinary(dataframe._jdf,location,compression,format)


class SpliceMLContext(PySpliceContext):
    """
    PySpliceContext for use with the cloud service.
    Although the original pysplicecontext *will work*
    on the Cloud Service (Zeppelin Notebook), this class
    does many things for ease of use.
    """
    @staticmethod
    def get_jdbc_url():
        """
        Get the JDBC Url for the current cluster (internal w/ no timeout)
        :return: (string) jdbc url
        """
        import os
        framework = os.environ['FRAMEWORK_NAME']
        return 'jdbc:splice://{framework}-proxy.marathon.mesos:1527/splicedb'.format(
            framework=framework)

    def __init__(self, sparkSession, useH2O=False, _unit_testing=False):
        """
        "Automagically" find the JDBC URL and establish a connection
        to the current Splice Machine database
        :param sparkSession: the sparksession object
        :param useH2O: whether or not to
        :param _unit_testing: whether or not we are unit testing
        """
        PySpliceContext.__init__(self, self.get_jdbc_url(), sparkSession, _unit_testing)

        if useH2O:
            from pysparkling import H2OConf, H2OContext
            h2oConf = H2OConf(sparkSession)
            h2oConf.set_fail_on_unsupported_spark_param_disabled()
            self.hc = H2OContext.getOrCreate(sparkSession, h2oConf)
