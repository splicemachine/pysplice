class FileExtensions:
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
    @staticmethod
    def map_from_mlflow_flavor(flavor: str):
        return {
            'spark': 'spark',
            'h2o': 'h2o',
            'keras': 'h5',
            'sklearn': 'pkl'
        }[flavor]

class DatabaseSupportedLibs:
    """
    Class containing supported model libraries for native database model deployment
    """
    @staticmethod
    def get_valid():
        return (
            'spark', 'h2o', 'sklearn', 'keras'
        )

class ModelStatuses:
    """
    Class containing names
    for In Database Model Deployments
    """
    deployed: str = 'DEPLOYED'
    deleted: str = 'DELETED'
    SUPPORTED_STATUSES = [deployed, deleted]

