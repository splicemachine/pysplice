import unittest
import time
from pyspark.sql import SparkSession
from splicemachine.spark.context import PySpliceContext
import uuid
import tempfile
import os


class PySpliceTest(unittest.TestCase):


    @classmethod
    def create_spark_session(cls):
        spark_session = SparkSession.builder.getOrCreate()
        #spark_session.sparkContext.setLogLevel("ERROR")
        logger = spark_session.sparkContext._jvm.org.apache.log4j
        logger.LogManager.getLogger("org").setLevel(logger.Level.OFF)
        logger.LogManager.getLogger("akka").setLevel(logger.Level.OFF)
        return spark_session

    @classmethod
    def create_testing_pysplice_session(cls,spark_session):
        db_url = 'jdbc:splice://localhost:1527/splicedb;user=splice;password=admin'
        splice_context = PySpliceContext(db_url,spark_session)
        return splice_context

    @classmethod
    def setUp(cls):
        cls.spark_session = cls.create_spark_session()
        cls.splice_context = cls.create_testing_pysplice_session(spark_session=cls.spark_session)


    @classmethod
    def tearDown(cls):
        cls.spark_session.stop()


class Test(PySpliceTest):
    def test_analyzeSchema(self):
        self.splice_context.analyzeSchema("splice")
        assert True


    def test_analyzeTable(self):
        self.splice_context.analyzeTable("sys.systables")
        assert True

    def test_executeUpdate(self):
        self.splice_context.executeUpdate("drop table if exists splice.systables")
        self.splice_context.executeUpdate("create table systables as select * from sys.systables")
        assert self.splice_context.tableExists("SPLICE.SYSTABLES")

    def test_dropTable(self):
        self.splice_context.executeUpdate("drop table if exists splice.pysplice_test_droptable")
        self.splice_context.executeUpdate("create table pysplice_test_droptable ( COL1 int primary key)")
        self.splice_context.dropTable("splice.pysplice_test_droptable")
        cnt = self.splice_context.df("select count(*) as cnt from sys.sysschemas a join sys.systables b on a.SCHEMAID = b.SCHEMAID where a.SCHEMANAME = 'SPLICE' and b.TABLENAME = 'PYSPLICE_TEST_DROPTABLE'").collect()[0]['CNT']
        assert cnt == 0

    def test_df(self):
        self.splice_context.executeUpdate("drop table if exists splice.pysplice_test_df")
        test_df_df = self.spark_session.createDataFrame([[1],[2]], "COL1: int")
        self.splice_context.executeUpdate("create table pysplice_test_df ( COL1 int primary key)")
        self.splice_context.insert(test_df_df,"splice.pysplice_test_df")
        cnt = self.splice_context.df("select count(*) as cnt from splice.pysplice_test_df").collect()[0]['CNT']
        assert cnt == 2

    def test_delete(self):
        self.splice_context.executeUpdate("drop table if exists splice.pysplice_test")
        test_delete_df = self.spark_session.createDataFrame([[1],[2]], "COL1: int")
        self.splice_context.executeUpdate("create table pysplice_test ( COL1 int primary key)")
        self.splice_context.insert(test_delete_df,"splice.pysplice_test")
        self.splice_context.delete(test_delete_df,"splice.pysplice_test")
        cnt = self.splice_context.df("select count(*) as cnt from splice.pysplice_test").collect()[0]['CNT']
        self.splice_context.dropTable("splice.pysplice_test")
        assert cnt == 0

    def test_execute(self):
        self.splice_context.execute("select count(*) from sys.systables")
        assert True

    def test_export(self):
        test_export_df = self.spark_session.createDataFrame([[1],[2]], "COL1:int")
        temp_dir = tempfile.gettempdir()
        file = os.path.join(temp_dir,str(uuid.uuid4()) + '.csv')
        print(file)
        self.splice_context.export(test_export_df,file)
        test_export_load_df = self.spark_session.read.option("timestampFormat", "yyyy/MM/dd HH:mm:ss ZZ").csv(file,inferSchema=True)
        assert test_export_df.count() == test_export_load_df.count()

    def test_exportBinary(self):
        test_exportBinary_df = self.spark_session.createDataFrame([[1],[2]], "COL1:int")
        temp_dir = tempfile.gettempdir()
        file = os.path.join(temp_dir,str(uuid.uuid4()) + '.parquet')
        self.splice_context.exportBinary(test_exportBinary_df,file,False,"parquet")
        load_df = self.spark_session.read.parquet(file)
        assert test_exportBinary_df.count() == load_df.count()

    def test_getSchema(self):
        systables_schema_from_df = self.splice_context.df("select * from sys.systables").schema
        systables_schema = self.splice_context.getSchema("sys.systables")
        assert systables_schema_from_df == systables_schema

    def test_insert(self):
        self.splice_context.executeUpdate("drop table if exists splice.pysplice_test_insert")
        test_insert_df = self.spark_session.createDataFrame([[1],[2]], "COL1 : int")
        self.splice_context.executeUpdate("create table splice.pysplice_test_insert ( col1 int primary key)")
        time.sleep(10)
        self.splice_context.insert(test_insert_df,"splice.pysplice_test_insert")
        cnt = self.splice_context.df("select count(*) as cnt from splice.pysplice_test_insert").collect()[0]['CNT']
        assert cnt == 2

    def test_internalDf(self):
        self.splice_context.executeUpdate("drop table if exists splice.pysplice_test")
        test_internalDf_df = self.spark_session.createDataFrame([[1],[2]], "COL1 : int")
        self.splice_context.executeUpdate("create table splice.pysplice_test ( col1 int primary key)")
        self.splice_context.insert(test_internalDf_df,"splice.pysplice_test")
        cnt = self.splice_context.internalDf("select count(*) as cnt from splice.pysplice_test").collect()[0]['CNT']
        assert cnt == 2

    def test_tableExists(self):
        self.splice_context.executeUpdate("drop table if exists splice.pysplice_test")
        self.splice_context.executeUpdate("create table pysplice_test ( COL1 int primary key)")
        return self.splice_context.tableExists("splice.pysplice_test")

    def test_truncateTable(self):
        self.splice_context.executeUpdate("drop table if exists splice.pysplice_test")
        test_truncateTable_df = self.spark_session.createDataFrame([[1],[2]], "COL1: int")
        self.splice_context.executeUpdate("create table pysplice_test ( col1 int primary key)")
        self.splice_context.insert(test_truncateTable_df,"splice.pysplice_test")
        self.splice_context.truncateTable("splice.pysplice_test")
        cnt = self.splice_context.df("select count(*) as cnt from splice.pysplice_test").collect()[0]['CNT']
        assert cnt == 0

    def test_update(self):
        test_update_df = self.spark_session.createDataFrame([[1,2],[2,3]], "COL1:int,COL2:int")
        test_update_update_df = self.spark_session.createDataFrame([[1,2],[2,4]],"COL1:int,COL2:int")
        self.splice_context.executeUpdate("drop table if exists splice.pysplice_test_update")
        self.splice_context.executeUpdate("create table pysplice_test_update ( COL1 int primary key,COL2 int)")
        self.splice_context.insert(test_update_df,"splice.pysplice_test_update")
        self.splice_context.update(test_update_update_df,"splice.pysplice_test_update")
        cnt = self.splice_context.df("select count(*) as cnt from splice.pysplice_test_update where col2 = 4").collect()[0]["CNT"]
        assert cnt == 1

    def test_upsert(self):
        test_upsert_df = self.spark_session.createDataFrame([[1,2],[2,3]], "COL1:int,COL2:int")
        test_upsert_upsert_df = self.spark_session.createDataFrame([[1,2],[2,4],[3,3]],"COL1:int,COL2:int")
        self.splice_context.executeUpdate("drop table if exists splice.pysplice_test_upsert")
        self.splice_context.executeUpdate("create table pysplice_test_upsert ( COL1 int primary key,COL2 int)")
        self.splice_context.insert(test_upsert_df,"splice.pysplice_test_upsert")
        self.splice_context.upsert(test_upsert_upsert_df,"splice.pysplice_test_upsert")
        cnt = self.splice_context.df("select count(*) as cnt from splice.pysplice_test_upsert where col2 = 2 or col1 = 3").collect()[0]["CNT"]
        assert cnt == 2
