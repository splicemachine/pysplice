from __future__ import print_function

from pyspark.sql import DataFrame
from py4j.java_gateway import java_import


class PySpliceContext:
    """
    This class implements a SpliceMachineContext object (similar to the SparkContext object)
    """

    def __init__(self, JDBC_URL, sparkSQLContext, _unitTesting=False):
        """
        :param JDBC_URL: (string) The JDBC URL Connection String for your Splice Machine Cluster
        :param sparkSQLContext: (sparkContext) A SparkContext Object for executing Spark Queries
        """
        self.jdbcurl = JDBC_URL
        self._unitTesting = _unitTesting

        if not _unitTesting:  # Private Internal Argument to Override Using JVM
            self.sparkSQLContext = sparkSQLContext
            self.jvm = self.sparkSQLContext._sc._jvm
            java_import(self.jvm, "com.splicemachine.spark.splicemachine.*")
            java_import(self.jvm, "org.apache.spark.sql.execution.datasources.jdbc.{JDBCOptions, JdbcUtils}")
            java_import(self.jvm, "scala.collection.JavaConverters._")
            self.context = self.jvm.com.splicemachine.spark.splicemachine.SplicemachineContext(self.jdbcurl)

        else:
            from .utils import FakeJContext
            self.sparkSQLContext = sparkSQLContext
            self.jvm = ''
            self.context = FakeJContext(self.jdbcurl)

    def getConnection(self):
        """
        Return a connection to the database
        """
        return self.context.getConnection()

    def tableExists(self, schemaTableName):
        """
        Check whether or not a table exists

        :param schemaTableName: (string) Table Name
        """
        return self.context.tableExists(schemaTableName)

    def dropTable(self, schemaTableName):  # works
        """
        Drop a specified table.

        :param schemaTableName (optional): (string) schemaName.tableName
        """
        return self.context.dropTable(schemaTableName)

    def df(self, sql):
        """
        Return a Spark Dataframe from the results of a Splice Machine SQL Query

        :param sql: (string) SQL Query (eg. SELECT * FROM table1 WHERE column2 > 3)
        :return: A Spark DataFrame containing the results
        """
        if self._unitTesting:
            return self.context.df(sql)
        return DataFrame(self.context.df(sql), self.sparkSQLContext)

    def insert(self, dataFrame, schemaTableName):
        """
        Insert a RDD into a table (schema.table).  The schema is required since RDD's do not have schema.

        :param dataFrame: (RDD) The dataFrame you would like to insert
        :param schemaTableName: (string) The table in which you would like to insert the RDD
        """
        return self.context.insert(dataFrame._jdf, schemaTableName)

    def delete(self, dataFrame, schemaTableName):
        """
        Delete records in a dataframe based on joining by primary keys from the data frame.
        Be careful with column naming and case sensitivity.

        :param dataFrame: (RDD) The dataFrame you would like to delete
        :param schemaTableName: (string) Splice Machine Table
        """
        return self.context.delete(dataFrame._jdf, schemaTableName)

    def update(self, dataFrame, schemaTableName):
        """
        Update data from a dataframe for a specified schemaTableName (schema.table).
        The keys are required for the update and any other columns provided will be updated in the rows.

        :param dataFrame: (RDD) The dataFrame you would like to update
        :param schemaTableName: (string) Splice Machine Table
        :return:
        """
        return self.context.update(dataFrame._jdf, schemaTableName)

    def getSchema(self, schemaTableName):
        """
        Return the schema via JDBC.

        :param schemaTableName: (string) Table name
        """
        return self.context.getSchema(schemaTableName)

