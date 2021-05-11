from pyspark.serializers import PickleSerializer, AutoBatchedSerializer
def _to_java_object_rdd(rdd):
    """ 
    Return a JavaRDD of Object by unpickling
    It will convert each Python object into Java object by Pyrolite, whenever the
    RDD is serialized in batch or not.
    
    :param rdd: The spark rdd
    """
    rdd = rdd._reserialize(AutoBatchedSerializer(PickleSerializer()))
    return rdd.ctx._jvm.org.apache.spark.mllib.api.python.SerDe.pythonToJava(rdd._jrdd, True)

def spark_df_size(df):
    """
    Returns the approximate size of a Spark DF in bytes
    
    :param df: The Spark Dataframe
    :param sc: An active Spark Context
    """
    JavaObj = _to_java_object_rdd(df.rdd)
    sizeof = df._sc._jvm.org.apache.spark.util.SizeEstimator.estimate(JavaObj)
    return sizeof    
