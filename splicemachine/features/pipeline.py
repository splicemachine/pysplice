from splicemachine import SpliceMachineException
from splicemachine.spark.context import ExtPySpliceContext
from splicemachine.features.constants import PipeLanguage

class Pipeline:
    def __init__(self, *, name, description, pipeline_start_date, pipeline_interval, pipeline_id=None, pipeline_version=None, pipes=None, feature_set_id=None, 
                    feature_set_version=None, pipeline_url=None, **kwargs):
        self.name = name.upper()
        self.description = description
        self.pipeline_start_date = pipeline_start_date
        self.pipeline_interval = pipeline_interval
        self.pipeline_id = pipeline_id
        self.pipeline_version = pipeline_version
        self.pipes = pipes
        self.feature_set_id = feature_set_id
        self.feature_set_version = feature_set_version
        self.pipeline_url = pipeline_url
        args = {k.lower(): kwargs[k] for k in kwargs}
        self.__dict__.update(args)

    def run(self):
        df = self.pipes[0].apply()
        last = self.pipes[0]
        for pipe in self.pipes[1:]:
            if last.language in [PipeLanguage.pyspark, PipeLanguage.sql] and pipe.language == PipeLanguage.python:
                df = df.toPandas()
            if last.language == PipeLanguage.python and pipe.language in [PipeLanguage.pyspark, PipeLanguage.sql]:
                df = pipe._pandas_to_spark(df)
            if pipe.language == PipeLanguage.sql:
                if isinstance(pipe.splice_ctx, ExtPySpliceContext):
                    raise SpliceMachineException(f'Error encountered with Pipe {pipe.name}: Cannot execute SQL statements on DataFrames '
                                                    'when using an ExtPySpliceContext. Please give your pipe a PySpliceContext with pipe.set_splice_ctx()')
                df = pipe._df_to_sql(df)
            df = pipe.apply(df)
            last = pipe
        return df
