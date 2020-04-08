from enum import Enum

# The MLeap package does not include a __version__ so we'll store it here
MLEAP_VERSION = '0.15.0'
# When storing models for in-DB deployment, we save the model with a name.
class DBLibraries():
    MLeap = 'mleap'
    H2OMOJO = 'h2omojo'
    SUPPORTED_LIBRARIES = [MLeap, H2OMOJO]

class H2OModelType(Enum): # Based off https://github.com/h2oai/h2o-3/blob/master/h2o-genmodel/src/main/java/hex/ModelCategory.java
    SINGULAR = 0 # Models that return a single value (Regression, HGLMRegression, Clustering)
    KEY_VALUE_RETURN = 1 # Models that only return N classes with values associated (Binomial, Multinomial, Ordinal)
    KNOWN_OUTPUTS = 2 # Models whose output labels are known (AutoEncoder, TargetEncoder, DimReduction, WordEmbedding, AnomalyDetection)


class SparkModelType(Enum):
    """
    Model types for MLeap Deployment to DB
    """
    CLASSIFICATION = 0
    REGRESSION = 1
    CLUSTERING_WITH_PROB = 2
    CLUSTERING_WO_PROB = 3
