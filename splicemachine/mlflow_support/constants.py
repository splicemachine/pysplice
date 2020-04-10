from enum import Enum

# The MLeap package does not include a __version__ so we'll store it here
MLEAP_VERSION = '0.15.0'
# When storing models for in-DB deployment, we save the model with a name.
class DBLibraries():
    MLeap = 'mleap'
    H2OMOJO = 'h2omojo'
    SUPPORTED_LIBRARIES = [MLeap, H2OMOJO]

class H2OModelType(Enum): # Based off https://github.com/h2oai/h2o-3/blob/master/h2o-genmodel/src/main/java/hex/ModelCategory.java
    REGRESSION = 0 # Models that return a single Double value (Regression, HGLMRegression)
    SINGULAR = 1 # Models that return a single Int value (Clustering)
    CLASSIFICATION = 2 # Models that only return N classes with values associated (Binomial, Multinomial, Ordinal)
    KEY_VALUE_RETURN = 3 # Models whose output labels are known (AutoEncoder, TargetEncoder, DimReduction, WordEmbedding, AnomalyDetection)


class SparkModelType(Enum):
    """
    Model types for MLeap Deployment to DB
    """
    CLASSIFICATION = 4
    REGRESSION = 5
    CLUSTERING_WITH_PROB = 6
    CLUSTERING_WO_PROB = 7
