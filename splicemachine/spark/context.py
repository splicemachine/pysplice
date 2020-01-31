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
        'StringType': 'VARCHAR(500)',
        'TimestampType': 'TIMESTAMP',
        'UnknownType': 'BLOB'
    }

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
            self.jvm = self.spark_sql_context._sc._jvm
            java_import(self.jvm, "com.splicemachine.spark.splicemachine.*")
            java_import(self.jvm,
                        "org.apache.spark.sql.execution.datasources.jdbc.{JDBCOptions, JdbcUtils}")
            java_import(self.jvm, "scala.collection.JavaConverters._")
            java_import(self.jvm, "com.splicemachine.derby.impl.*")
            self.jvm.com.splicemachine.derby.impl.SpliceSpark.setContext(
                self.spark_sql_context._jsc)
            self.context = self.jvm.com.splicemachine.spark.splicemachine.SplicemachineContext(
                self.jdbcurl)

        else:
            from .tests.mocked import MockedScalaContext
            self.spark_sql_context = sparkSession._wrapped
            self.jvm = ''
            self.context = MockedScalaContext(self.jdbcurl)
            
    def toUpper(self, dataframe):
        """
        Returns a dataframe with all of the columns in uppercase
        :param dataframe: The dataframe to convert to uppercase
        """
        for col in dataframe.columns:
            dataframe = dataframe.withColumnRenamed(col, col.upper())
        return dataframe

    def replaceDataframeSchema(self, dataframe, schema_table_name):
        """
        Returns a dataframe with all column names replaced with the proper string case from the DB table
        :param dataframe: A dataframe with column names to convert
        :param schema_table_name: The schema.table with the correct column cases to pull from the database
        """
        cols = self.df('select top 1 * from {}'.format(schema_table_name)).columns
        #sort the columns case insensitive so we are replacing the right ones in the dataframe
        old_cols = sorted(dataframe.columns, key=lambda s: s.lower())
        new_cols = sorted(cols, key=lambda s: s.lower())
        for old_col,new_col in zip(old_cols,new_cols):
            dataframe = dataframe.withColumnRenamed(old_col, new_col)
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
        # make sure column names are in the correct case
        dataframe = self.replaceDataframeSchema(dataframe, schema_table_name)
        return self.context.insert(dataframe._jdf, schema_table_name)

    def upsert(self, dataframe, schema_table_name):
        """
        Upsert the data from a dataframe into a table (schema.table).

        :param dataframe: (DF) The dataframe you would like to upsert
        :param schema_table_name: (string) The table in which you would like to upsert the RDD
        """
        # make sure column names are in the correct case
        dataframe = self.replaceDataframeSchema(dataframe, schema_table_name)
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
        """
        # make sure column names are in the correct case
        dataframe = self.replaceDataframeSchema(dataframe, schema_table_name)
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
        if self._unit_testing:
            return self.context.internalDf(query_string)
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

    def export(self, dataframe, location, compression=False, replicationCount=1, fileEncoding=None,
               fieldSeparator=None,
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
        return self.context.export(dataframe._jdf, location, compression, replicationCount,
                                   fileEncoding,
                                   fieldSeparator, quoteCharacter)

    def exportBinary(self, dataframe, location, compression, e_format):
        '''
        Export a dataFrame in binary format
        :param dataframe:
        :param location: Destination directory
        :param compression: Whether to compress the output or not
        :param e_format: Binary format to be used, currently only 'parquet' is supported
        :return:
        '''
        return self.context.exportBinary(dataframe._jdf, location, compression, e_format)

    def _generateDBSchema(self, dataframe, types={}):
        """
        Generate the schema for create table
        """
        #convert keys and values to uppercase in the types dictionary
        types = dict((key.upper(), val) for key,val in types.items())
        db_schema = []
        #convert dataframe to have all uppercase column names
        dataframe = self.toUpper(dataframe)
        #i contains the name and pyspark datatype of the column
        for i in dataframe.schema:
            if i.name.upper() in types:
                print('Column {} is of type {}'.format(i.name.upper(),i.dataType))
                dt = types[i.name.upper()]
            else:
                dt = PySpliceContext.CONVERSIONS[str(i.dataType)]
            db_schema.append((i.name.upper(),dt))
        
        return db_schema
    
    def _getCreateTableSchema(self, schema_table_name, new_schema=False):
        """
        Parse schema for new table; if it is needed,
        create it
        """
        #try to get schema and table, else set schema to splice
        if '.' in schema_table_name:
            schema, table = schema_table_name.upper().split('.')
        else:
            schema = self.getConnection().getCurrentSchemaName()
            table = schema_table_name.upper()
        #check for new schema
        if new_schema:
            print('Creating schema {}'.format(schema))
            self.execute('CREATE SCHEMA {}'.format(schema))
        
        return schema, table
    
    def _dropTableIfExists(self, schema, table):
        """
        Drop table if it exists
        """
        print('Creating table {schema}.{table}'.format(schema=schema,table=table))
        self.execute('DROP TABLE IF EXISTS {schema}.{table}'.format(schema=schema,table=table))
    
    def createTable(self, dataframe, schema_table_name, new_schema=False, drop_table=False, types = {}):
        '''
        Creates a schema.table from a dataframe
        :param schema_table_name: String full table name in the format "schema.table_name"
                                  If only a table name is provided (ie no '.' in the string) schema SPLICE will be assumed
                                  If this table exists in the database already, it will be DROPPED and a new one will be created
        :param dataframe: The dataframe that the table will be created for
        :param new_schema: A boolean to create a new schema. If True, the function will create a new schema before creating the table. If the schema already exists, set to False [DEFAULT True]
        :param drop_table: An optinal boolean to drop the table if it exists. [DEFAULT False]
        :param types: An optional dictionary of type {string: string} containing column names and their respective SQL types. The values of the dictionary MUST be valid SQL types. See https://doc.splicemachine.com/sqlref_datatypes_intro.html
            If None or if any types are missing, types will be assumed automatically from the dataframe schema as follows:
                    BooleanType: BOOLEAN
                    ByteType: TINYINT
                    DateType: DATE
                    DoubleType: DOUBLE
                    DecimalType: DOUBLE
                    IntegerType: INTEGER
                    LongType: BIGINT
                    NullType: VARCHAR(50)
                    ShortType: SMALLINT
                    StringType: VARCHAR(150)
                    TimestampType: TIMESTAMP
                    UnknownType: BLOB
        '''
        db_schema = self._generateDBSchema(dataframe, types=types)
        schema, table = self._getCreateTableSchema(schema_table_name, new_schema=new_schema)
        # Make sure table doesn't exists already
        if(not drop_table and self.tableExists(schema_table_name)):
           return('ERROR: Table already exists. Please drop it or set drop_table option to True')
           
        self._dropTableIfExists(schema,table)
        sql = 'CREATE TABLE {schema}.{table}(\n'.format(schema=schema,table=table)
        for name,typ in db_schema:
            sql += '{} {},\n'.format(name,typ)
        sql = sql[:-2] + ')'
        print(sql)
        self.execute(sql)


class SpliceMLContext(object):
    def __init__(self):
        raise Exception("This class has been deprecated in favor of the PySpliceContext class. "
                        "the JDBC URL argument in the constructor is now *optional*. Thus, if "
                        "running on the cloud service, you could do this "
                        "`splice=PySpliceContext(spark)` to achieve the same result")
