from typing import List, Dict, Optional, Tuple
from pyspark.sql.dataframe import DataFrame as SparkDF
from splicemachine.spark import PySpliceContext
from datetime import datetime

TIMESTAMP_FORMAT = 'yyyy-MM-dd HH:mm:ss'

def clean_df(df, cols):
    for old,new in zip(df.columns, cols):
        df = df.withColumnRenamed(old,new)
    return df

class FeatureStore:
    def __init__(self, splice_ctx: PySpliceContext, mlflow_ctx = None):
        self.splice_ctx = splice_ctx
        self.mlflow_ctx = mlflow_ctx

    def register_splice_context(self, splice_ctx):
        self.splice_ctx = splice_ctx

    def register_mlflow_context(self, mlflow_ctx):
        self.mlflow_ctx = mlflow_ctx

    def get_feature_sets(self, feature_set_ids: List[int] = None, _filter: Dict[str,str] = None) -> List[FeatureSet]:
        """
        Returns a list of available feature sets

        :return: List[FeatureSet]
        """
        feature_sets = []
        feature_set_ids = feature_set_ids or []
        _filter = _filter or {}

        sql = '''
        SELECT fset.FeatureSetID, TableName, SchemaName, Description, pkcolumns FROM FeatureStore.FeatureSet fset
        INNER JOIN 
        (SELECT FeatureSetID, STRING_AGG(KeyColumnName,',') PKColumns FROM FeatureStore.FeatureSetKey GROUP BY 1)  p ON fset.FeatureSetID=p.FeatureSetID 
        '''

        # Filter by featuresetid and filter
        if feature_set_ids or _filter:
            sql += ' WHERE '
        if feature_set_ids:
            fsd = tuple(feature_set_ids) if len(feature_set_ids) > 1 else f'({feature_set_ids[0]})'
            sql += f' fset.featuresetID in {fsd} AND'
        for fl in _filter:
            sql += f" fset.{fl}='{_filter[fl]}' AND"
        sql = sql.rstrip('AND')

        feature_set_rows = self.splice_ctx.df(sql)
        cols = ['FeatureSetID', 'TableName', 'SchemaName', 'Description', 'pkcolumns']
        feature_set_rows = clean_df(feature_set_rows, cols)

        for fs in feature_set_rows.collect():
            d = fs.asDict()
            feature_sets.append(FeatureSet(splice, **d))
        return feature_sets

    def get_training_contexts(self, _filter: Dict[str,str] = None) -> List[TrainingContext]:
        """
        Returns all available training contexts in the format of a dictionary mapping
        Context_ID: (context_name, context_description)
        :param filter: Dictionary container the filter keyword (label, description etc) and the value to filter on (using CONTAINS)

        :return:
        """
        training_contexts = []

        sql = '''
              SELECT tc.ContextID, tc.Name, tc.Description, CAST(SQLText AS VARCHAR(1000)) context_sql, 
                   p.PKColumns, 
                   TSColumn, LabelColumn,
                   c.ContextColumns               
              FROM FeatureStore.TrainingContext tc 
                   INNER JOIN 
                    (SELECT ContextID, STRING_AGG(KeyColumnName,',') PKColumns FROM FeatureStore.TrainingContextKey WHERE KeyType='P' GROUP BY 1)  p ON tc.ContextID=p.ContextID 
                   INNER JOIN 
                    (SELECT ContextID, STRING_AGG(KeyColumnName,',') ContextColumns FROM FeatureStore.TrainingContextKey WHERE KeyType='C' GROUP BY 1)  c ON tc.ContextID=c.ContextID 
              '''

        if _filter:
            sql += ' WHERE '
            for k in _filter:
                sql += f"tc.{k}='{_filter[k]}' and"
            sql = sql.rstrip('and')

        training_context_rows = self.splice_ctx.df(sql)

        cols = ['contextid','name','description','context_sql','pkcolumns','tscolumn','labelcolumn','contextcolumns']

        training_context_rows = clean_df(training_context_rows,cols)

        for tc in training_context_rows.collect():
            t = tc.asDict()
            training_contexts.append(TrainingContext(**t))
        return training_contexts


    def run_feature_elimination(self, df, label: str = 'label', n: int = 10, verbose: int = 0,
                                log_mlflow: bool = False, mlflow_run_name: str = None) -> None:
        # TODO: Webinar
        """
        Runs feature elimination using a Spark decision tree on the dataframe passed in. Optionally logs results to mlflow
        :param df: The dataframe with features and label
        :param label: the label column
        :param n: The number of features desired. Default 10
        :param verbose: The level of verbosity. 0 indicated no printing. 1 indicates printing remaining features after
            each round. 2 indicates print features and relative importances after each round. Default 0
        :param log_mlflow: Whether or not to log results to mlflow as nested runs. Default false
        :param mlflow_run_name: The name of the parent run under which all subsequent runs will live. The children run
            names will be {mlflow_run_name}_{num_features}_features. ie testrun_5_features, testrun_4_features etc
        :return:
        """
    def get_training_context_id(self, name) -> int:
        """
        Returns the unique context ID from a name

        :param name:
        :return:
        """


    def add_feature_set(self, feature_set: FeatureSet):
        """
        Add a feature set

        :param ft:
        :return:
        """
        self.feature_sets.append(feature_set)

    def create_feature_set(self, schema: str, name: str, pk_columns: Dict[str,str],
                             feature_column: Dict[str,str], desc: Optional[str] = None) -> FeatureSet:
        # TODO: Webinar
        """
        Creates a new feature set, recording metadata and generating the database table with associated triggers

        :param schema:
        :param name:
        :param pk_columns:
        :param feature_column:
        :param desc:
        :return: FeatureTable
        """
        pass

    def get_features_by_name(self, name: List[str]) -> List[Feature]:
        pass

    def create_training_context(self, *, name: str, sql: str, primary_keys: List[str],
                            ts_col: str, label_col: Optional[str] = None, replace: Optional[bool] = False,
                            context_keys: List[str] = None, desc: Optional[str] = None) -> None:
        # TODO: Webinar
        """
        Registers a training context for use in generating training SQL

        :param name: The training set name. This must be unique to other existing training sets unless replace is True
        :param sql: (str) a SELECT statement that includes:
            * the primary key column(s) - uniquely identifying a training row/case
            * the inference timestamp column - timestamp column with which to join features (temporal join timestamp)
            * context key(s) - the references to the other feature tables' primary keys (ie customer_id, location_id)
            * (optionally) the label expression - defining what the training set is trying to predict
        :param primary_keys: (List[str]) The list of columns from the training SQL that identify the training row
        :param ts_col: (Optional[str]) The timestamp column of the training SQL that identifies the inference timestamp
        :param label_col: (Optional[str]) The optional label column from the training SQL.
        :param replace: (Optional[bool]) Whether to replace an existing training set
        :param context_keys: (List[str]) A list of context keys in the sql that are used to get the desired features in get_training_set
            Note: If this parameter is None, default to all other columns in the sql that aren't PK, label, ts etc
        :param desc: (Optional[str]) An optional description of the training set
        :return:
        """

        # validate_sql()
        # register_training_context()
        pass

    def get_feature_context_keys(self, features: List[str]) -> Dict[str,List[str]]:
        """
        Returns a dictionary mapping each individual feature to its primary key(s)

        :param features: (List[str]) The list of features to get primary keys for
        :return: Dict[str, List[str]]
        """
        pass

    def get_available_features(self, training_context_id: int) -> List[Feature]:
        """
        Given a training context ID or name, returns the available features

        :param training_context:
        :return:
        """
        df = splice.df(f'''
        SELECT f.FEATUREID, f.FEATURESETID, f.NAME, f.DESCRIPTION, f.FEATUREDATATYPE, f.FEATURETYPE, f.CARDINALITY, f.TAGS, f.COMPLIANCELEVEL, f.LASTUPDATETS, f.LASTUPDATEUSERID
          FROM FeatureStore.Feature f
          WHERE FeatureID IN
          (
              SELECT f.FeatureID FROM
                FeatureStore.TrainingContext tc 
                INNER JOIN 
                FeatureStore.TrainingContextKey c ON c.ContextID=tc.ContextID AND c.KeyType='C'
                INNER JOIN 
                FeatureStore.FeatureSetKey fsk ON c.KeyColumnName=fsk.KeyColumnName
                INNER JOIN
                FeatureStore.Feature f USING (FeatureSetID)
              WHERE tc.ContextID={training_context_id}
          )
        ''')
        cols = ['FEATUREID', 'FEATURESETID', 'NAME', 'DESCRIPTION', 'FEATUREDATATYPE', 'FEATURETYPE', 'CARDINALITY','TAGS', 'COMPLIANCELEVEL', 'LASTUPDATETS', 'LASTUPDATEUSERID']
        df = clean_df(df, cols)

        features = []
        for feat in df.collect():
            f = feat.asDict()
            features.append(Feature(**f))
        return features


    def set_feature_description(self):
        pass

    def get_feature_description(self):
        pass

    def get_training_set(self, training_context_id: int, features: List[Feature], start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None, return_sql: bool = False) -> SparkDF or str:
        # TODO: Webinar
        """
        Returns the training set as a Spark Dataframe

        :param training_context_id: (int) The name of the registered training context
        :param features: (List[str]) the list of features from the feature store to be included in the training
            * NOTE: This function will error if the context SQL is missing a context key required to retrieve the\
             desired features
        :param start_time: (Optional[datetime]) The start time of the query (how far back in the data to start). Default None
            * NOTE: If start_time is None, query will start from beginning of history
        :param end_time: (Optional[datetime]) The end time of the query (how far recent in the data to get). Default None
            * NOTE: If end_time is None, query will get most recently available data
        :param return_sql: (Optional[bool]) Return the SQL statement (str) instead of the Spark DF. Defaults False
        :return: Optional[SparkDF, str]
        """

        # DB-9556 loss of column names on complex sql for NSDS
        cols = []

        # Get training context information (ctx primary key column(s), ctx primary key inference ts column, )
        tctx = self.get_training_contexts(_filter={'CONTEXTID': training_context_id})[0]
        # SELECT clause
        sql = 'SELECT '
        for pkcol in tctx.pkcolumns: # Select primary key column(s)
            sql += f'ctx.{pkcol},'
            cols.append(pkcol)

        sql += f' ctx.{tctx.tscolumn}, ' # Select timestamp column
        cols.append(tctx.tscolumn)

        for feature in features:
            sql += f'COALESCE(fset{feature.featuresetid}.{feature.name},fset{feature.featuresetid}h.{feature.name}) {feature.name},' # Collect all features over time
            cols.append(feature.name)

        sql = sql + f'ctx.{tctx.labelcolumn}' if tctx.labelcolumn else sql.rstrip(',')  # Select the optional label col
        if tctx.labelcolumn: cols.append(tctx.labelcolumn)

        # FROM clause
        sql += f' FROM ({tctx.context_sql}) ctx '

        # JOIN clause
        feature_set_ids = list({f.featuresetid for f in features}) # Distinct set of IDs
        feature_sets = self.get_feature_sets(feature_set_ids)
        for fset in feature_sets:
            # Join Feature Set
            sql += f' LEFT OUTER JOIN {fset.schemaname}.{fset.tablename} fset{fset.featuresetid} ON '
            for pkcol in fset.pkcolumns:
                sql += f' fset{fset.featuresetid}.{pkcol}=ctx.{pkcol} AND '
            sql += f' ctx.{tctx.tscolumn} >= fset{fset.featuresetid}.LAST_UPDATE_TS '

            # Join Feature Set History
            sql += f' LEFT OUTER JOIN {fset.schemaname}.{fset.tablename}_history fset{fset.featuresetid}h ON '
            for pkcol in fset.pkcolumns:
                sql += f' fset{fset.featuresetid}h.{pkcol}=ctx.{pkcol} AND '
            sql += f' ctx.{tctx.tscolumn} >= fset{fset.featuresetid}h.ASOF_TS AND ctx.{tctx.tscolumn} < fset{fset.featuresetid}h.UNTIL_TS'

        # WHERE clause on optional start and end times
        if start_time or end_time:
            sql += ' WHERE '
            if start_time:
                sql += f"ctx.{tctx.tscolumn} >= '{str(start_time)}' AND"
            if end_time:
                sql += f"ctx.{tctx.tscolumn} <= '{str(end_time)}'"
            sql = sql.rstrip('AND')

        return sql if return_sql else clean_df(splice.df(sql),cols)

    def list_training_sets(self) -> Dict[str, Optional[str]]:
        """
        Returns a dictionary a training sets available, with the map name -> description. If there is no description,
        the value will be an emtpy string

        :return: Dict[str, Optional[str]]
        """
        pass

