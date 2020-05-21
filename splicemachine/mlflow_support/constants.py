from enum import Enum

# When storing models for in-DB deployment, we save the model with a name.
class DBLibraries():
    MLeap = 'mleap'
    H2OMOJO = 'h2omojo'
    SKLearn = 'sklearn'
    Keras = 'keras'
    SUPPORTED_LIBRARIES = [MLeap, H2OMOJO, SKLearn, Keras]

# DB_LIB_UTILS: dict = {
#     DBLibraries.MLeap: SparkUtils,
#     DBLibraries.Keras: KerasUtils,
#     DBLibraries.SKLearn: SKUtils,
#     DBLibraries.H2OMOJO: H2OUtils
# }

class H2OModelType(Enum): # Based off https://github.com/h2oai/h2o-3/blob/master/h2o-genmodel/src/main/java/hex/ModelCategory.java
    REGRESSION = 0 # Models that return a single Double value (Regression, HGLMRegression)
    SINGULAR = 1 # Models that return a single Int value (Clustering)
    CLASSIFICATION = 2 # Models that only return N classes with values associated (Binomial, Multinomial, Ordinal)
    KEY_VALUE = 3 # Models whose output labels are known (AutoEncoder, TargetEncoder, DimReduction, WordEmbedding, AnomalyDetection)


class SparkModelType(Enum):
    """
    Model types for MLeap Deployment to DB
    """
    CLASSIFICATION = 0
    REGRESSION = 1
    CLUSTERING_WITH_PROB = 2
    CLUSTERING_WO_PROB = 3

class SklearnModelType(Enum):
    """
    Model Types for SKLearn models
    Sklearn isn't as well defined in their model categories, so we are going to classify them by their return values
    """
    POINT_PREDICTION_REG = 0
    POINT_PREDICTION_CLF = 1
    KEY_VALUE = 2

class KerasModelType(Enum):
    """
    Model Types for SKLearn models
    Sklearn isn't as well defined in their model categories, so we are going to classify them by their return values
    """
    REGRESSION = 0
    KEY_VALUE = 1


class FileExtensions():
    """
    Class containing names for
    valid File Extensions
    """
    spark: str = "spark"
    keras: str = "h5"
    h2o: str = "h2o"
    sklearn: str = "pkl"

    @staticmethod
    def get_valid() -> tuple:
        """
        Return a tuple of the valid file extensions
        in Database
        :return: (tuple) valid statuses
        """
        return (
            FileExtensions.spark, FileExtensions.keras, FileExtensions.h2o, FileExtensions.sklearn
        )
