"""
Copyright 2021 Splice Machine, Inc.

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
import re
from string import punctuation as bad_chars

from py4j.java_gateway import java_import
from pyspark.sql import DataFrame
from pyspark.sql.types import _parse_datatype_json_string, StringType

from splicemachine.spark.constants import CONVERSIONS
from splicemachine import SpliceMachineException


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
            java_import(
                self.jvm, "org.apache.spark.sql.execution.datasources.jdbc.JDBCOptions")
            java_import(
                self.jvm, "org.apache.spark.sql.execution.datasources.jdbc.JdbcUtils")
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

    def columnNamesCaseSensitive(self, caseSensitive):
        """
        Sets whether column names should be treated as case sensitive.

        :param caseSensitive: (boolean) True for case sensitive, False for not case sensitive
        """
        self.context.columnNamesCaseSensitive(caseSensitive)

    def toUpper(self, dataframe):
        """
        Returns a dataframe with all of the columns in uppercase

        :param dataframe: (Dataframe) The dataframe to convert to uppercase
        """
        for s in dataframe.schema:
            s.name = s.name.upper()
        # You need to re-generate the dataframe for the capital letters to take effect
        return dataframe.rdd.toDF(dataframe.schema)

    def toLower(self, dataframe):
        """
        Returns a dataframe with all of the columns in lowercase

        :param dataframe: (Dataframe) The dataframe to convert to lowercase
        """
        for s in dataframe.schema:
            s.name = s.name.lower()
        # You need to re-generate the dataframe for the capital letters to take effect
        return dataframe.rdd.toDF(dataframe.schema)


    def replaceDataframeSchema(self, dataframe, schema_table_name):
        """
        Returns a dataframe with all column names replaced with the proper string case from the DB table

        :param dataframe: (Dataframe) A dataframe with column names to convert
        :param schema_table_name: (str) The schema.table with the correct column cases to pull from the database
        :return: (DataFrame) A Spark DataFrame with the replaced schema
        """
        schema = self.getSchema(schema_table_name)
        # Fastest way to replace the column case if changed
        dataframe = dataframe.rdd.toDF(schema)
        return dataframe

    def fileToTable(self, file_path, schema_table_name, primary_keys=None, drop_table=False, **pandas_args):
        """
        Load a file from the local filesystem or from a remote location and create a new table
        (or recreate an existing table), and load the data from the file into the new table. Any file_path that can be
        read by pandas should work here.

        :param file_path: The local file to load
        :param schema_table_name: The schema.table name
        :param primary_keys: List[str] of primary keys for the table. Default None
        :param drop_table: Whether or not to drop the table. If this is False and the table already exists, the
            function will fail. Default False
        :param pandas_args: Extra parameters to be passed into the pd.read_csv function. Any parameters accepted
            in pd.read_csv will work here
        :return: None
        """
        import pandas as pd
        pdf = pd.read_csv(file_path, **pandas_args)
        df = self.pandasToSpark(pdf)
        self.createTable(df, schema_table_name, primary_keys=primary_keys, drop_table=drop_table, to_upper=True)
        self.insert(df, schema_table_name, to_upper=True)

    def pandasToSpark(self, pdf):
        """
        Convert a Pandas DF to Spark, and try to manage NANs from Pandas in case of failure. Spark cannot handle
        Pandas NAN existing in String columns (as it considers it NaN Number ironically), so we replace the occurances
        with a temporary value and then convert it back to null after it becomes a Spark DF

        :param pdf: The Pandas dataframe
        :return: The Spark DF
        """
        try: # Try to create the dataframe as it exists
            return self.spark_session.createDataFrame(pdf)
        except TypeError:
            p_df = pdf.copy()
            # This means there was an NaN conversion error
            from pyspark.sql.functions import udf
            for c in p_df.columns: # Replace non numeric/time columns with a custom null value
                if p_df[c].dtype not in ('int64','float64', 'datetime64[ns]'):
                    p_df[c].fillna('Splice_Temp_NA', inplace=True)
            spark_df = self.spark_session.createDataFrame(p_df)
            # Convert that custom null value back to null after converting to a spark dataframe
            null_replace_udf = udf(lambda name: None if name == "Splice_Temp_NA" else name, StringType())
            for field in spark_df.schema:
                if field.dataType==StringType():
                    spark_df = spark_df.withColumn(field.name, null_replace_udf(spark_df[field.name]))
                spark_df = spark_df.withColumnRenamed(field.name, re.sub(r'['+bad_chars+' ]', '_',field.name))
            # Replace NaN numeric columns with null
            spark_df = spark_df.replace(float('nan'), None)
            return spark_df


    def getConnection(self):
        """
        Return a connection to the database
        """
        return self.context.getConnection()

    def tableExists(self, schema_and_or_table_name, table_name=None):
        """
        Check whether or not a table exists

        :Example:
            .. code-block:: python

                splice.tableExists('schemaName.tableName')\n
                # or\n
                splice.tableExists('schemaName', 'tableName')

        :param schema_and_or_table_name: (str) Pass the schema name in this param when passing the table_name param,
          or pass schemaName.tableName in this param without passing the table_name param
        :param table_name: (optional) (str) Table Name, used when schema_and_or_table_name contains only the schema name
        :return: (bool) whether or not the table exists
        """
        if table_name:
            return self.context.tableExists(schema_and_or_table_name, table_name)
        else:
            return self.context.tableExists(schema_and_or_table_name)

    def dropTable(self, schema_and_or_table_name, table_name=None):
        """
        Drop a specified table.

        :Example:
            .. code-block:: python

                splice.dropTable('schemaName.tableName') \n
                # or\n
                splice.dropTable('schemaName', 'tableName')

        :param schema_and_or_table_name: (str) Pass the schema name in this param when passing the table_name param,
          or pass schemaName.tableName in this param without passing the table_name param
        :param table_name: (optional) (str) Table Name, used when schema_and_or_table_name contains only the schema name
        :return: None
        """
        if table_name:
            self.context.dropTable(schema_and_or_table_name, table_name)
        else:
            self.context.dropTable(schema_and_or_table_name)

    def df(self, sql, to_lower=False):
        """
        Return a Spark Dataframe from the results of a Splice Machine SQL Query

        :Example:
            .. code-block:: python

                df = splice.df('SELECT * FROM MYSCHEMA.TABLE1 WHERE COL2 > 3')

        :param sql: (str) SQL Query (eg. SELECT * FROM table1 WHERE col2 > 3)
        :param to_lower: Whether or not to convert column names from the dataframe to lowercase
        :return: (Dataframe) A Spark DataFrame containing the results
        """
        df = DataFrame(self.context.df(sql), self.spark_sql_context)
        return self.toLower(df) if to_lower else df

    def insert(self, dataframe, schema_table_name, to_upper=True, create_table=False):
        """
        Insert a dataframe into a table (schema.table).

        :param dataframe: (Dataframe) The dataframe you would like to insert
        :param schema_table_name: (str) The table in which you would like to insert the DF
        :param to_upper: (bool) If the dataframe columns should be converted to uppercase before table creation
                            If False, the table will be created with lower case columns. [Default True]
        :param create_table: If the table does not exists at the time of the call, the table will first be created
        :return: None
        """
        if to_upper:
            dataframe = self.toUpper(dataframe)
        if not self.tableExists(schema_table_name):
            if not create_table:
                raise SpliceMachineException("Table does not exist. Create the table first or set create_table=True "
                                             "in this function, or call createAndInsertTable")
            else:
                print('Table does not yet exist, creating table... ',end='')
                self.createTable(dataframe, schema_table_name, to_upper=to_upper)
                print('Done.')
        self.context.insert(dataframe._jdf, schema_table_name)

    def insertWithStatus(self, dataframe, schema_table_name, statusDirectory, badRecordsAllowed, to_upper=True):
        """
        Insert a dataframe into a table (schema.table) while tracking and limiting records that fail to insert.
        The status directory and number of badRecordsAllowed allow for duplicate primary keys to be
        written to a bad records file.  If badRecordsAllowed is set to -1, all bad records will be written
        to the status directory.

        :param dataframe: (Dataframe) The dataframe you would like to insert
        :param schema_table_name: (str) The table in which you would like to insert the dataframe
        :param statusDirectory: (str) The status directory where bad records file will be created
        :param badRecordsAllowed: (int) The number of bad records are allowed. -1 for unlimited
        :param to_upper: Whether to convert dataframe columns to uppercase before action. Default true
        :return: None
        """
        if to_upper:
            dataframe = self.toUpper(dataframe)
        self.context.insert(dataframe._jdf, schema_table_name, statusDirectory, badRecordsAllowed)

    def insertRdd(self, rdd, schema, schema_table_name):
        """
        Insert an rdd into a table (schema.table)

        :param rdd: (RDD) The RDD you would like to insert
        :param schema: (StructType) The schema of the rows in the RDD
        :param schema_table_name: (str) The table in which you would like to insert the RDD
        :return: None
        """
        self.insert(
            self.createDataFrame(rdd, schema),
            schema_table_name
        )

    def insertRddWithStatus(self, rdd, schema, schema_table_name, statusDirectory, badRecordsAllowed):
        """
        Insert an rdd into a table (schema.table) while tracking and limiting records that fail to insert. \
        The status directory and number of badRecordsAllowed allow for duplicate primary keys to be \
        written to a bad records file.  If badRecordsAllowed is set to -1, all bad records will be written \
        to the status directory.

        :param rdd: (RDD) The RDD you would like to insert
        :param schema: (StructType) The schema of the rows in the RDD
        :param schema_table_name: (str) The table in which you would like to insert the dataframe
        :param statusDirectory: (str) The status directory where bad records file will be created
        :param badRecordsAllowed: (int) The number of bad records are allowed. -1 for unlimited
        :return: None
        """
        self.insertWithStatus(
            self.createDataFrame(rdd, schema),
            schema_table_name,
            statusDirectory,
            badRecordsAllowed
        )

    def upsert(self, dataframe, schema_table_name, to_upper=True):
        """
        Upsert the data from a dataframe into a table (schema.table).
        If triggers fail when calling upsert, use the mergeInto function instead of upsert.

        :param dataframe: (Dataframe) The dataframe you would like to upsert
        :param schema_table_name: (str) The table in which you would like to upsert the RDD
        :param to_upper: Whether to convert dataframe columns to uppercase before action. Default true
        :return: None
        """
        # make sure column names are in the correct case
        if to_upper:
            dataframe = self.toUpper(dataframe)
        self.context.upsert(dataframe._jdf, schema_table_name)

    def upsertWithRdd(self, rdd, schema, schema_table_name):
        """
        Upsert the data from an RDD into a table (schema.table).
        If triggers fail when calling upsertWithRdd, use the mergeIntoWithRdd function instead of upsertWithRdd.

        :param rdd: (RDD) The RDD you would like to upsert
        :param schema: (StructType) The schema of the rows in the RDD
        :param schema_table_name: (str) The table in which you would like to upsert the RDD
        :return: None
        """
        self.upsert(
            self.createDataFrame(rdd, schema),
            schema_table_name
        )

    def mergeInto(self, dataframe, schema_table_name, to_upper=True):
        """
        Rows in the dataframe whose primary key is not in schemaTableName will be inserted into the table;
        rows in the dataframe whose primary key is in schemaTableName will be used to update the table.

        This implementation differs from upsert in a way that allows triggers to work.

        :param dataframe: (Dataframe) The dataframe you would like to merge in
        :param schema_table_name: (str) The table in which you would like to merge in the dataframe
        :param to_upper: Whether to convert dataframe columns to uppercase before action. Default true
        :return: None
        """
        if to_upper:
            dataframe = self.toUpper(dataframe)
        self.context.mergeInto(dataframe._jdf, schema_table_name)

    def mergeIntoWithRdd(self, rdd, schema, schema_table_name):
        """
        Rows in the rdd whose primary key is not in schemaTableName will be inserted into the table;
        rows in the rdd whose primary key is in schemaTableName will be used to update the table.

        This implementation differs from upsertWithRdd in a way that allows triggers to work.

        :param rdd: (RDD) The RDD you would like to merge in
        :param schema: (StructType) The schema of the rows in the RDD
        :param schema_table_name: (str) The table in which you would like to merge in the RDD
        :return: None
        """
        self.mergeInto(
            self.createDataFrame(rdd, schema),
            schema_table_name
        )

    def delete(self, dataframe, schema_table_name):
        """
        Delete records in a dataframe based on joining by primary keys from the data frame.
        Be careful with column naming and case sensitivity.

        :param dataframe: (Dataframe) The dataframe you would like to delete
        :param schema_table_name: (str) Splice Machine Table
        :return: None
        """
        self.context.delete(dataframe._jdf, schema_table_name)

    def deleteWithRdd(self, rdd, schema, schema_table_name):
        """
        Delete records using an rdd based on joining by primary keys from the rdd.
        Be careful with column naming and case sensitivity.

        :param rdd: (RDD) The RDD containing the primary keys you would like to delete from the table
        :param schema: (StructType) The schema of the rows in the RDD
        :param schema_table_name: (str) Splice Machine Table
        :return: None
        """
        self.delete(
            self.createDataFrame(rdd, schema),
            schema_table_name
        )

    def update(self, dataframe, schema_table_name, to_upper=True):
        """
        Update data from a dataframe for a specified schema_table_name (schema.table).
        The keys are required for the update and any other columns provided will be updated
        in the rows.

        :param dataframe: (Dataframe) The dataframe you would like to update
        :param schema_table_name: (str) Splice Machine Table
        :param to_upper: Whether to convert dataframe columns to uppercase before action. Default true
        """
        # make sure column names are in the correct case
        if to_upper:
            dataframe = self.toUpper(dataframe)
        self.context.update(dataframe._jdf, schema_table_name)

    def updateWithRdd(self, rdd, schema, schema_table_name):
        """
        Update data from an rdd for a specified schema_table_name (schema.table).
        The keys are required for the update and any other columns provided will be updated
        in the rows.

        :param rdd: (RDD) The RDD you would like to use for updating the table
        :param schema: (StructType) The schema of the rows in the RDD
        :param schema_table_name: (str) Splice Machine Table
        :return: None
        """
        self.update(
            self.createDataFrame(rdd, schema),
            schema_table_name
        )

    def getSchema(self, schema_table_name):
        """
        Return the schema via JDBC.

        :param schema_table_name: (str) Table name
        :return: (StructType) PySpark StructType representation of the table
        """
        return _parse_datatype_json_string(self.context.getSchema(schema_table_name).json())

    def execute(self, query_string):
        '''
        execute a query over JDBC

        :Example:
            .. code-block:: python
            
                splice.execute('DELETE FROM TABLE1 WHERE col2 > 3')

        :param query_string: (str) SQL Query (eg. SELECT * FROM table1 WHERE col2 > 3)
        :return: None
        '''
        self.context.execute(query_string)

    def executeUpdate(self, query_string):
        '''
        execute a dml query:(update,delete,drop,etc)

        :Example:
            .. code-block:: python

                splice.executeUpdate('DROP TABLE table1')

        :param query_string: (string) SQL Query (eg. DROP TABLE table1)
        :return: None
        '''
        self.context.executeUpdate(query_string)

    def internalDf(self, query_string):
        '''
        SQL to Dataframe translation (Lazy). Runs the query inside Splice Machine and sends the results to the Spark Adapter app

        :param query_string: (str) SQL Query (eg. SELECT * FROM table1 WHERE col2 > 3)
        :return: (DataFrame) pyspark dataframe contains the result of query_string
        '''
        return DataFrame(self.context.internalDf(query_string), self.spark_sql_context)

    def rdd(self, schema_table_name, column_projection=None):
        """
        Table with projections in Splice mapped to an RDD.

        :param schema_table_name: (string) Accessed table
        :param column_projection: (list of strings) Names of selected columns
        :return: (RDD[Row]) the result of the projection
        """
        if column_projection:
            colnames = ', '.join(str(col) for col in column_projection)
        else:
            colnames = '*'
        return self.df('select '+colnames+' from '+schema_table_name).rdd

    def internalRdd(self, schema_table_name, column_projection=None):
        """
        Table with projections in Splice mapped to an RDD.
        Runs the projection inside Splice Machine and sends the results to the Spark Adapter app as an rdd

        :param schema_table_name: (str) Accessed table
        :param column_projection: (list of strings) Names of selected columns
        :return: (RDD[Row]) the result of the projection
        """
        if column_projection:
            colnames = ', '.join(str(col) for col in column_projection)
        else:
            colnames = '*'
        return self.internalDf('select '+colnames+' from '+schema_table_name).rdd

    def truncateTable(self, schema_table_name):
        """
        Truncate a table

        :param schema_table_name: (str) the full table name in the format "schema.table_name" which will be truncated
        :return: None
        """
        self.context.truncateTable(schema_table_name)

    def analyzeSchema(self, schema_name):
        """
        Analyze the schema

        :param schema_name: (str) schema name which stats info will be collected
        :return: None
        """
        self.context.analyzeSchema(schema_name)

    def analyzeTable(self, schema_table_name, estimateStatistics=False, samplePercent=10.0):
        """
        Collect stats info on a table
        
        :param schema_table_name: full table name in the format of 'schema.table'
        :param estimateStatistics: will use estimate statistics if True
        :param samplePercent: the percentage or rows to be sampled.
        :return: None
        """
        self.context.analyzeTable(schema_table_name, estimateStatistics, float(samplePercent))

    def export(self,
               dataframe,
               location,
               compression=False,
               replicationCount=1,
               fileEncoding=None,
               fieldSeparator=None,
               quoteCharacter=None):
        """
        Export a dataFrame in CSV

        :param dataframe: (DataFrame)
        :param location: (str) Destination directory
        :param compression: (bool) Whether to compress the output or not
        :param replicationCount: (int) Replication used for HDFS write
        :param fileEncoding: (str) fileEncoding or None, defaults to UTF-8
        :param fieldSeparator: (str) fieldSeparator or None, defaults to ','
        :param quoteCharacter: (str) quoteCharacter or None, defaults to '"'
        :return: None
        """
        self.context.export(dataframe._jdf, location, compression, replicationCount,
                                   fileEncoding, fieldSeparator, quoteCharacter)

    def exportBinary(self, dataframe, location, compression, e_format='parquet'):
        """
        Export a dataFrame in binary format

        :param dataframe: (DataFrame)
        :param location: (str) Destination directory
        :param compression: (bool) Whether to compress the output or not
        :param e_format: (str) Binary format to be used, currently only 'parquet' is supported. [Default 'parquet']
        :return: None
        """
        self.context.exportBinary(dataframe._jdf, location, compression, e_format)

    def bulkImportHFile(self, dataframe, schema_table_name, options):
        """
        Bulk Import HFile from a dataframe into a schema.table

        :param dataframe: (DataFrame)
        :param schema_table_name: (str) Full table name in the format of "schema.table"
        :param options: (Dict) Dictionary of options to be passed to --splice-properties; bulkImportDirectory is required
        :return: (int) Number of records imported
        """
        optionsMap = self.jvm.java.util.HashMap()
        for k, v in options.items():
            optionsMap.put(k, v)
        return self.context.bulkImportHFile(dataframe._jdf, schema_table_name, optionsMap)

    def bulkImportHFileWithRdd(self, rdd, schema, schema_table_name, options):
        """
        Bulk Import HFile from an rdd into a schema.table

        :param rdd: (RDD) Input data
        :param schema: (StructType) The schema of the rows in the RDD
        :param schema_table_name: (str) Full table name in the format of "schema.table"
        :param options: (Dict) Dictionary of options to be passed to --splice-properties; bulkImportDirectory is required
        :return:  (int) Number of records imported
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

        :param dataframe: (DataFrame) Input data
        :param schema_table_name: (str) Full table name in the format of "schema.table"
        :param sample_fraction: (float) A value between 0 and 1 that specifies the percentage of data in the dataFrame \
        that should be sampled to determine the splits. \
        For example, specify 0.005 if you want 0.5% of the data sampled.
        :return: None
        """
        self.context.splitAndInsert(dataframe._jdf, schema_table_name, float(sample_fraction))

    def createDataFrame(self, rdd, schema):
        """
        Creates a dataframe from a given rdd and schema.

        :param rdd: (RDD) Input data
        :param schema: (StructType) The schema of the rows in the RDD
        :return: (DataFrame) The Spark DataFrame
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
                print('Column {} is of type {}'.format(
                    i.name.upper(), i.dataType))
                dt = types[i.name.upper()]
            else:
                dt = CONVERSIONS[str(i.dataType)]
            db_schema.append((i.name.upper(), dt))

        return db_schema

    def _getCreateTableSchema(self, schema_table_name, new_schema=False):
        """
        Parse schema for new table; if it is needed, create it
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

    def _dropTableIfExists(self, schema_table_name, table_name=None):
        """
        Drop table if it exists
        """
        if self.tableExists(schema_and_or_table_name=schema_table_name, table_name=table_name):
            print('Table exists. Dropping table')
            self.dropTable(schema_and_or_table_name=schema_table_name, table_name=table_name)

    def dropTableIfExists(self, schema_table_name, table_name=None):
        """
        Drops a table if exists
        
        :Example:
            .. code-block:: python

                splice.dropTableIfExists('schemaName.tableName') \n
                # or\n
                splice.dropTableIfExists('schemaName', 'tableName')

        :param schema_table_name: (str) Pass the schema name in this param when passing the table_name param,
          or pass schemaName.tableName in this param without passing the table_name param
        :param table_name: (optional) (str) Table Name, used when schema_table_name contains only the schema name
        :return: None
        """
        self._dropTableIfExists(schema_table_name, table_name)

    def _jstructtype(self, schema):
        """
        Convert python StructType to java StructType

        :param schema: PySpark StructType
        :return: Java Spark StructType
        """
        return self.spark_session._jsparkSession.parseDataType(schema.json())

    def createTable(self, dataframe, schema_table_name, primary_keys=None,
                    create_table_options=None, to_upper=True, drop_table=False):
        """
        Creates a schema.table (schema_table_name) from a dataframe
        
        :param dataframe: The Spark DataFrame to base the table off
        :param schema_table_name: str The schema.table to create
        :param primary_keys: List[str] the primary keys. Default None
        :param create_table_options: str The additional table-level SQL options default None
        :param to_upper: bool If the dataframe columns should be converted to uppercase before table creation. \
            If False, the table will be created with lower case columns. Default True
        :param drop_table: bool whether to drop the table if it exists. Default False. If False and the table exists, the function will throw an exception
        :return: None

        """
        if drop_table:
            self._dropTableIfExists(schema_table_name)
        if to_upper:
            dataframe = self.toUpper(dataframe)
        primary_keys = primary_keys if primary_keys else []
        self.createTableWithSchema(schema_table_name, dataframe.schema,
                                   keys=primary_keys, create_table_options=create_table_options)

    def createTableWithSchema(self, schema_table_name, schema, keys=None, create_table_options=None):
        """
        Creates a schema.table from a schema

        :param schema_table_name: str The schema.table to create
        :param schema: (StructType) The schema that describes the columns of the table
        :param keys: (List[str]) The primary keys. Default None
        :param create_table_options: (str) The additional table-level SQL options. Default None
        :return: None
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

    def createAndInsertTable(self, dataframe, schema_table_name, primary_keys=None,
                             create_table_options=None, to_upper=True):
        """
        Creates a schema.table (schema_table_name) from a dataframe and inserts the dataframe into the table

        :param dataframe: The Spark DataFrame to base the table off
        :param schema_table_name: str The schema.table to create
        :param primary_keys: List[str] the primary keys. Default None
        :param create_table_options: str The additional table-level SQL options default None
        :param to_upper: bool If the dataframe columns should be converted to uppercase before table creation. \
            If False, the table will be created with lower case columns. Default True
        :param drop_table: bool whether to drop the table if it exists. Default False. If False and the table exists, the function will throw an exception
        :return: None

        """
        if self.tableExists(schema_table_name):
            raise SpliceMachineException(f'Table {schema_table_name} already exists. Drop the table first or call '
                                         f'splice.insert with the provided dataframe')
        self.createTable(dataframe, schema_table_name, primary_keys=primary_keys,
                                 create_table_options=create_table_options, to_upper=to_upper)
        self.insert(dataframe, schema_table_name, to_upper=to_upper)

class ExtPySpliceContext(PySpliceContext):
    """
    This class implements a SplicemachineContext object from com.splicemachine.spark2 for use outside of the K8s Cloud Service
    """
    _spliceSparkPackagesName = "com.splicemachine.spark2.splicemachine.*"

    def _splicemachineContext(self):
        return self.jvm.com.splicemachine.spark2.splicemachine.SplicemachineContext(
            self.jdbcurl, self.kafkaServers, self.kafkaPollTimeout)

    def __init__(self, sparkSession, JDBC_URL=None, kafkaServers='localhost:9092',
                 kafkaPollTimeout=20000, _unit_testing=False):
        """
        :param JDBC_URL: (string) The JDBC URL Connection String for your Splice Machine Cluster
        :param sparkSession: (sparkContext) A SparkSession object for talking to Spark
        :param kafkaServers (string) Comma-separated list of Kafka broker addresses in the form host:port
        :param kafkaPollTimeout (int) Number of milliseconds to wait when polling Kafka
        """
        self.kafkaServers = kafkaServers
        self.kafkaPollTimeout = kafkaPollTimeout
        super().__init__(sparkSession, JDBC_URL, _unit_testing)

    def setAutoCommitOn(self):
        """
        Turn auto-commit on.  Auto-commit is on by default when the class is instantiated.

        :return: None
        """
        self.context.setAutoCommitOn()

    def setAutoCommitOff(self):
        """
        Turn auto-commit off.

        :return: None
        """
        self.context.setAutoCommitOff()

    def autoCommitting(self):
        """
        Check whether auto-commit is on.

        :return: (Boolean) True if auto-commit is on.
        """
        return self.context.autoCommitting()

    def transactional(self):
        """
        Check whether auto-commit is off.

        :return: (Boolean) True if auto-commit is off.
        """
        return self.context.transactional()

    def commit(self):
        """
        Commit the transaction.  Throws exception if auto-commit is on.

        :return: None
        """
        self.context.commit()

    def rollback(self):
        """
        Rollback the transaction.  Throws exception if auto-commit is on.

        :return: None
        """
        self.context.rollback()

    def rollbackToSavepoint(self, savepoint):
        """
        Rollback to the savepoint.  Throws exception if auto-commit is on.
        :param savepoint: (java.sql.Savepoint) A Savepoint.

        :return: None
        """
        self.context.rollback(savepoint)

    def setSavepoint(self):
        """
        Create and set a unnamed savepoint at the current point in the transaction.  Throws exception if auto-commit is on.

        :return: (java.sql.Savepoint) The unnamed Savepoint
        """
        return self.context.setSavepoint()

    def setSavepointWithName(self, name):
        """
        Create and set a named savepoint at the current point in the transaction.  Throws exception if auto-commit is on.
        :param name: (String) The name of the Savepoint.

        :return: (java.sql.Savepoint) The named Savepoint
        """
        return self.context.setSavepoint(name)

    def releaseSavepoint(self, savepoint):
        """
        Release the savepoint.  Throws exception if auto-commit is on.
        :param savepoint: (java.sql.Savepoint) A Savepoint.

        :return: None
        """
        self.context.releaseSavepoint(savepoint)
