"""
Copyright 2020 Splice Machine, Inc.

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
from pyspark.sql.types import _parse_datatype_json_string
from splicemachine.spark.constants import CONVERSIONS


class PySpliceContext:
    """
    This class implements a SpliceMachineContext object (similar to the SparkContext object)
    """
    _spliceSparkPackagesName = "com.splicemachine.spark.splicemachine.*"

    def _splicemachineContext(self):
        return self.jvm.com.splicemachine.spark.splicemachine.SplicemachineContext(self.jdbcurl)

    def __init__(self, sparkSession, JDBC_URL=None, _unit_testing=False):
        """
        :param JDBC_URL: (string) The JDBC URL Connection String for your Splice Machine Cluster
        :param sparkSession: (sparkContext) A SparkSession object for talking to Spark
        """

        if JDBC_URL:
            self.jdbcurl = JDBC_URL
        else:
            try:
                self.jdbcurl = os.environ['BEAKERX_SQL_DEFAULT_JDBC']
            except KeyError as e:
                raise KeyError(
                    "Could not locate JDBC URL. If you are not running on the cloud service,"
                    "please specify the JDBC_URL=<some url> keyword argument in the constructor"
                )

        self._unit_testing = _unit_testing

        if not _unit_testing:  # Private Internal Argument to Override Using JVM
            self.spark_sql_context = sparkSession._wrapped
            self.spark_session = sparkSession
            self.jvm = self.spark_sql_context._sc._jvm
            java_import(self.jvm, self._spliceSparkPackagesName)
            java_import(self.jvm,
                        "org.apache.spark.sql.execution.datasources.jdbc.{JDBCOptions, JdbcUtils}")
            java_import(self.jvm, "scala.collection.JavaConverters._")
            java_import(self.jvm, "com.splicemachine.derby.impl.*")
            java_import(self.jvm, 'org.apache.spark.api.python.PythonUtils')
            self.jvm.com.splicemachine.derby.impl.SpliceSpark.setContext(
                self.spark_sql_context._jsc)
            self.context = self._splicemachineContext()

        else:
            from .tests.mocked import MockedScalaContext
            self.spark_sql_context = sparkSession._wrapped
            self.spark_session = sparkSession
            self.jvm = ''
            self.context = MockedScalaContext(self.jdbcurl)

    def toUpper(self, dataframe):
        """
        Returns a dataframe with all of the columns in uppercase
        :param dataframe: The dataframe to convert to uppercase
        """
        for s in dataframe.schema:
            s.name = s.name.upper() # Modifying the schema automatically modifies the Dataframe (passed by reference)
        return dataframe


    def replaceDataframeSchema(self, dataframe, schema_table_name):
        """
        Returns a dataframe with all column names replaced with the proper string case from the DB table
        :param dataframe: A dataframe with column names to convert
        :param schema_table_name: The schema.table with the correct column cases to pull from the database
        """
        schema = self.getSchema(schema_table_name)
        dataframe = dataframe.rdd.toDF(schema) #Fastest way to replace the column case if changed
        return dataframe

    def getConnection(self):
        """
        Return a connection to the database
        """
        return self.context.getConnection()

    def tableExists(self, schema_and_or_table_name, table_name=None):
        """
        Check whether or not a table exists

        Call it like:
            tableExists('schemaName.tableName')
        Or:
            tableExists('schemaName', 'tableName')

        :param schema_and_or_table_name: (string) Pass the schema name in this param when passing the table_name param,
          or pass schemaName.tableName in this param without passing the table_name param
        :param table_name: (optional) (string) Table Name, used when schema_and_or_table_name contains only the schema name
        """
        if table_name:
            return self.context.tableExists(schema_and_or_table_name, table_name)
        else:
            return self.context.tableExists(schema_and_or_table_name)

    def dropTable(self, schema_and_or_table_name, table_name=None):
        """
        Drop a specified table.

        Call it like:
            dropTable('schemaName.tableName')
        Or:
            dropTable('schemaName', 'tableName')

        :param schema_and_or_table_name: (string) Pass the schema name in this param when passing the table_name param,
          or pass schemaName.tableName in this param without passing the table_name param
        :param table_name: (optional) (string) Table Name, used when schema_and_or_table_name contains only the schema name
        """
        if table_name:
            return self.context.dropTable(schema_and_or_table_name, table_name)
        else:
            return self.context.dropTable(schema_and_or_table_name)

    def df(self, sql):
        """
        Return a Spark Dataframe from the results of a Splice Machine SQL Query

        :param sql: (string) SQL Query (eg. SELECT * FROM table1 WHERE column2 > 3)
        :return: A Spark DataFrame containing the results
        """
        return DataFrame(self.context.df(sql), self.spark_sql_context)

    def insert(self, dataframe, schema_table_name):
        """
        Insert a dataframe into a table (schema.table).

        :param dataframe: (DF) The dataframe you would like to insert
        :param schema_table_name: (string) The table in which you would like to insert the dataframe
        """
        # make sure column names are in the correct case
        dataframe = self.replaceDataframeSchema(dataframe, schema_table_name)
        return self.context.insert(dataframe._jdf, schema_table_name)

    def insertWithStatus(self, dataframe, schema_table_name, statusDirectory, badRecordsAllowed):
        """
        Insert a dataframe into a table (schema.table) while tracking and limiting records that fail to insert.
        The status directory and number of badRecordsAllowed allow for duplicate primary keys to be
        written to a bad records file.  If badRecordsAllowed is set to -1, all bad records will be written
        to the status directory.

        :param dataframe: (DF) The dataframe you would like to insert
        :param schema_table_name: (string) The table in which you would like to insert the dataframe
        :param statusDirectory The status directory where bad records file will be created
        :param badRecordsAllowed The number of bad records are allowed. -1 for unlimited
        """
        dataframe = self.replaceDataframeSchema(dataframe, schema_table_name)
        return self.context.insert(dataframe._jdf, schema_table_name, statusDirectory, badRecordsAllowed)

    def insertRdd(self, rdd, schema, schema_table_name):
        """
        Insert an rdd into a table (schema.table).

        :param rdd: (RDD) The RDD you would like to insert
        :param schema: (StructType) The schema of the rows in the RDD
        :param schema_table_name: (string) The table in which you would like to insert the RDD
        """
        return self.insert(
            self.createDataFrame(rdd, schema),
            schema_table_name
        )

    def insertRddWithStatus(self, rdd, schema, schema_table_name, statusDirectory, badRecordsAllowed):
        """
        Insert an rdd into a table (schema.table) while tracking and limiting records that fail to insert.
        The status directory and number of badRecordsAllowed allow for duplicate primary keys to be
        written to a bad records file.  If badRecordsAllowed is set to -1, all bad records will be written
        to the status directory.

        :param rdd: (RDD) The RDD you would like to insert
        :param schema: (StructType) The schema of the rows in the RDD
        :param schema_table_name: (string) The table in which you would like to insert the dataframe
        :param statusDirectory The status directory where bad records file will be created
        :param badRecordsAllowed The number of bad records are allowed. -1 for unlimited
        """
        return self.insertWithStatus(
            self.createDataFrame(rdd, schema),
            schema_table_name,
            statusDirectory,
            badRecordsAllowed
        )

    def upsert(self, dataframe, schema_table_name):
        """
        Upsert the data from a dataframe into a table (schema.table).

        :param dataframe: (DF) The dataframe you would like to upsert
        :param schema_table_name: (string) The table in which you would like to upsert the RDD
        """
        # make sure column names are in the correct case
        dataframe = self.replaceDataframeSchema(dataframe, schema_table_name)
        return self.context.upsert(dataframe._jdf, schema_table_name)

    def upsertWithRdd(self, rdd, schema, schema_table_name):
        """
        Upsert the data from an RDD into a table (schema.table).

        :param rdd: (RDD) The RDD you would like to upsert
        :param schema: (StructType) The schema of the rows in the RDD
        :param schema_table_name: (string) The table in which you would like to upsert the RDD
        """
        return self.upsert(
            self.createDataFrame(rdd, schema),
            schema_table_name
        )

    def delete(self, dataframe, schema_table_name):
        """
        Delete records in a dataframe based on joining by primary keys from the data frame.
        Be careful with column naming and case sensitivity.

        :param dataframe: (DF) The dataframe you would like to delete
        :param schema_table_name: (string) Splice Machine Table
        """
        return self.context.delete(dataframe._jdf, schema_table_name)

    def deleteWithRdd(self, rdd, schema, schema_table_name):
        """
        Delete records using an rdd based on joining by primary keys from the rdd.
        Be careful with column naming and case sensitivity.

        :param rdd: (RDD) The RDD containing the primary keys you would like to delete from the table
        :param schema: (StructType) The schema of the rows in the RDD
        :param schema_table_name: (string) Splice Machine Table
        """
        return self.delete(
            self.createDataFrame(rdd, schema),
            schema_table_name
        )

    def update(self, dataframe, schema_table_name):
        """
        Update data from a dataframe for a specified schema_table_name (schema.table).
        The keys are required for the update and any other columns provided will be updated
        in the rows.

        :param dataframe: (DF) The dataframe you would like to update
        :param schema_table_name: (string) Splice Machine Table
        """
        # make sure column names are in the correct case
        dataframe = self.replaceDataframeSchema(dataframe, schema_table_name)
        return self.context.update(dataframe._jdf, schema_table_name)

    def updateWithRdd(self, rdd, schema, schema_table_name):
        """
        Update data from an rdd for a specified schema_table_name (schema.table).
        The keys are required for the update and any other columns provided will be updated
        in the rows.

        :param rdd: (RDD) The RDD you would like to use for updating the table
        :param schema: (StructType) The schema of the rows in the RDD
        :param schema_table_name: (string) Splice Machine Table
        """
        return self.update(
            self.createDataFrame(rdd, schema),
            schema_table_name
        )

    def getSchema(self, schema_table_name):
        """
        Return the schema via JDBC.

        :param schema_table_name: (DF) Table name
        """
        return _parse_datatype_json_string(self.context.getSchema(schema_table_name).json())

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

    def rdd(self, schema_table_name, column_projection=None):
        """
        Table with projections in Splice mapped to an RDD.
        :param schema_table_name: (string) Accessed table
        :param column_projection: (list of strings) Names of selected columns
        :return RDD[Row] with the result of the projection
        """
        if column_projection:
            colnames = ', '.join(str(col) for col in column_projection)
        else:
            colnames = '*'
        return self.df('select '+colnames+' from '+schema_table_name).rdd

    def internalRdd(self, schema_table_name, column_projection=None):
        """
        Table with projections in Splice mapped to an RDD.
        :param schema_table_name: (string) Accessed table
        :param column_projection: (list of strings) Names of selected columns
        :return RDD[Row] with the result of the projection
        """
        if column_projection:
            colnames = ', '.join(str(col) for col in column_projection)
        else:
            colnames = '*'
        return self.internalDf('select '+colnames+' from '+schema_table_name).rdd

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

    def analyzeTable(self, schema_table_name, estimateStatistics=False, samplePercent=10.0):
        """
        collect stats info on a table
        :param schema_table_name: full table name in the format of "schema.table"
        :param estimateStatistics:will use estimate statistics if True
        :param samplePercent:  the percentage or rows to be sampled.
        :return:
        """
        return self.context.analyzeTable(schema_table_name, estimateStatistics, float(samplePercent))

    def export(self, dataframe, location, compression=False, replicationCount=1, fileEncoding=None,
               fieldSeparator=None,
               quoteCharacter=None):
        """
        Export a dataFrame in CSV
        :param dataframe:
        :param location: Destination directory
        :param compression: Whether to compress the output or not
        :param replicationCount:  Replication used for HDFS write
        :param fileEncoding: fileEncoding or null, defaults to UTF-8
        :param fieldSeparator: fieldSeparator or null, defaults to ','
        :param quoteCharacter: quoteCharacter or null, defaults to '"'
        :return:
        """
        return self.context.export(dataframe._jdf, location, compression, replicationCount,
                                   fileEncoding,
                                   fieldSeparator, quoteCharacter)

    def exportBinary(self, dataframe, location, compression, e_format):
        """
        Export a dataFrame in binary format
        :param dataframe:
        :param location: Destination directory
        :param compression: Whether to compress the output or not
        :param e_format: Binary format to be used, currently only 'parquet' is supported
        :return:
        """
        return self.context.exportBinary(dataframe._jdf, location, compression, e_format)

    def bulkImportHFile(self, dataframe, schema_table_name, options):
        """
        Bulk Import HFile from a dataframe into a schema.table
        :param dataframe: Input data
        :param schema_table_name: Full table name in the format of "schema.table"
        :param options: Dictionary of options to be passed to --splice-properties; bulkImportDirectory is required
        """
        optionsMap = self.jvm.java.util.HashMap()
        for k, v in options.items():
            optionsMap.put(k, v)
        return self.context.bulkImportHFile(dataframe._jdf, schema_table_name, optionsMap)

    def bulkImportHFileWithRdd(self, rdd, schema, schema_table_name, options):
        """
        Bulk Import HFile from an rdd into a schema.table
        :param rdd: Input data
        :param schema: (StructType) The schema of the rows in the RDD
        :param schema_table_name: Full table name in the format of "schema.table"
        :param options: Dictionary of options to be passed to --splice-properties; bulkImportDirectory is required
        """
        return self.bulkImportHFile(
            self.createDataFrame(rdd, schema),
            schema_table_name,
            options
        )

    def splitAndInsert(self, dataframe, schema_table_name, sample_fraction):
        """
        Sample the dataframe, split the table, and insert a dataFrame into a schema.table.
        This corresponds to an insert into from select statement
        :param dataframe: Input data
        :param schema_table_name: Full table name in the format of "schema.table"
        :param sample_fraction: (float) A value between 0 and 1 that specifies the percentage of data in the dataFrame
            that should be sampled to determine the splits.
            For example, specify 0.005 if you want 0.5% of the data sampled.
        """
        return self.context.splitAndInsert(dataframe._jdf, schema_table_name, float(sample_fraction))

    def createDataFrame(self, rdd, schema):
        """
        Creates a dataframe from a given rdd and schema.

        :param rdd: Input data
        :param schema: (StructType) The schema of the rows in the RDD
        """
        return self.spark_session.createDataFrame(rdd, schema)

    def _generateDBSchema(self, dataframe, types={}):
        """
        Generate the schema for create table
        """
        # convert keys and values to uppercase in the types dictionary
        types = dict((key.upper(), val) for key, val in types.items())
        db_schema = []
        # convert dataframe to have all uppercase column names
        dataframe = self.toUpper(dataframe)
        # i contains the name and pyspark datatype of the column
        for i in dataframe.schema:
            if i.name.upper() in types:
                print('Column {} is of type {}'.format(i.name.upper(), i.dataType))
                dt = types[i.name.upper()]
            else:
                dt = CONVERSIONS[str(i.dataType)]
            db_schema.append((i.name.upper(), dt))

        return db_schema

    def _getCreateTableSchema(self, schema_table_name, new_schema=False):
        """
        Parse schema for new table; if it is needed,
        create it
        """
        # try to get schema and table, else set schema to splice
        if '.' in schema_table_name:
            schema, table = schema_table_name.upper().split('.')
        else:
            schema = self.getConnection().getCurrentSchemaName()
            table = schema_table_name.upper()
        # check for new schema
        if new_schema:
            print('Creating schema {}'.format(schema))
            self.execute('CREATE SCHEMA {}'.format(schema))

        return schema, table

    def _dropTableIfExists(self, schema_table_name):
        """
        Drop table if it exists
        """
        print('Dropping table {schema}.{table}'.format(schema=schema, table=table))
        self.execute('DROP TABLE IF EXISTS {schema}.{table}'.format(schema=schema, table=table))

    def _jstructtype(self, schema):
        """
        Convert python StructType to java StructType
        """
        return self.spark_session._jsparkSession.parseDataType(schema.json())

    def createTable(self, schema_table_name, dataframe, keys=None, create_table_options=None, to_upper=False):
        """
        Creates a schema.table from a dataframe
        :param schema_table_name: str The schema.table to create
        :param dataframe: The Spark DataFrame to base the table off
        :param keys: List[str] the primary keys. Default None
        :param create_table_options: str The additional table-level SQL options default None
        :param to_upper: bool If the dataframe columns should be converted to uppercase before table creation
                            If False, the table will be created with lower case columns. Default False
        """
        if to_upper:
            dataframe = self.toUpper(dataframe)
        self.context.createTableWithSchema(schema_table_name, df.schema, keys=keys, create_table_options=create_table_options)
        
    def createTableWithSchema(self, schema_table_name, schema, keys=None, create_table_options=None):
        """
        Creates a schema.table from a schema
        :param schema_table_name: str The schema.table to create
        :param schema: (StructType) The schema that describes the columns of the table
        :param keys: List[str] The primary keys. Default None
        :param create_table_options: str The additional table-level SQL options. Default None
        """
        if keys:
            keys_seq = self.jvm.PythonUtils.toSeq(keys)
        else:
            keys_seq = self.jvm.PythonUtils.toSeq([])
        self.context.createTable(
            schema_table_name,
            self._jstructtype(schema),
            keys_seq,
            create_table_options
        )

class ExtPySpliceContext(PySpliceContext):
    """
    This class implements a SplicemachineContext object from com.splicemachine.spark2
    """
    _spliceSparkPackagesName = "com.splicemachine.spark2.splicemachine.*"

    def _splicemachineContext(self):
        return self.jvm.com.splicemachine.spark2.splicemachine.SplicemachineContext(
            self.jdbcurl, self.kafkaServers, self.kafkaPollTimeout)

    def __init__(self, sparkSession, JDBC_URL=None, kafkaServers='localhost:9092', kafkaPollTimeout=20000, _unit_testing=False):
        """
        :param JDBC_URL: (string) The JDBC URL Connection String for your Splice Machine Cluster
        :param sparkSession: (sparkContext) A SparkSession object for talking to Spark
        :param kafkaServers (string) Comma-separated list of Kafka broker addresses in the form host:port
        :param kafkaPollTimeout (int) Number of milliseconds to wait when polling Kafka
        """
        self.kafkaServers = kafkaServers
        self.kafkaPollTimeout = kafkaPollTimeout
        super().__init__(sparkSession, JDBC_URL, _unit_testing)

