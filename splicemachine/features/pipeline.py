class Pipeline:
    def __init__(self, *, name, description, pipeline_start_ts, pipeline_interval, pipeline_id=None, pipeline_version=None, pipes=None, feature_set_id=None, 
                    feature_set_version=None, pipeline_url=None, **kwargs):
        self.name = name.upper()
        self.description = description
        self.pipeline_start_ts = pipeline_start_ts
        self.pipeline_interval = pipeline_interval
        self.pipeline_id = pipeline_id
        self.pipeline_version = pipeline_version
        self.pipes = pipes
        self.feature_set_id = feature_set_id
        self.feature_set_version = feature_set_version
        self.pipeline_url = pipeline_url
        args = {k.lower(): kwargs[k] for k in kwargs}
        self.__dict__.update(args)

    def run_pipeline(self, df):
        for pipe in self.pipes:
            df = pipe.apply(df)
        return df
