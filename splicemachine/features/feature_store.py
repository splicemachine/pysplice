from typing import List, Dict, Optional, Union
from pyspark.sql.dataframe import DataFrame as SparkDF
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import StringIndexer, VectorAssembler
from .constants import SQL, Columns
from .training_context import TrainingContext
from .utils import clean_df
from IPython.display import display
import pandas as pd
from splicemachine.spark import PySpliceContext
from datetime import datetime
from splicemachine.features import Feature, FeatureSet


class FeatureStore:
    def __init__(self, splice_ctx: PySpliceContext):
        self.splice_ctx = splice_ctx
        # self.feature_sets = self.get_feature_sets()

    def register_splice_context(self, splice_ctx):
        self.splice_ctx = splice_ctx

    def get_feature_sets(self, feature_set_ids: List[int] = None, _filter: Dict[str,str] = None) -> List[FeatureSet]:
        """
        Returns a list of available feature sets

        :return: List[FeatureSet]
        """
        feature_sets = []
        feature_set_ids = feature_set_ids or []
        _filter = _filter or {}

        sql = SQL.get_feature_sets

        # Filter by feature_set_id and filter
        if feature_set_ids or _filter:
            sql += ' WHERE '
        if feature_set_ids:
            fsd = tuple(feature_set_ids) if len(feature_set_ids) > 1 else f'({feature_set_ids[0]})'
            sql += f' fset.feature_set_id in {fsd} AND'
        for fl in _filter:
            sql += f" fset.{fl}='{_filter[fl]}' AND"
        sql = sql.rstrip('AND')

        feature_set_rows = self.splice_ctx.df(sql)
        cols = Columns.feature_set
        feature_set_rows = clean_df(feature_set_rows, cols)

        for fs in feature_set_rows.collect():
            d = fs.asDict()
            pkcols = d.pop('pk_columns').split('|')
            pktypes = d.pop('pk_types').split('|')
            d['primary_keys'] = {c:k for c,k in zip(pkcols,pktypes)}
            feature_sets.append(FeatureSet(splice_ctx=self.splice_ctx, **d))
        return feature_sets

    def remove_feature_set(self):
        # TODO
        pass

    def get_training_contexts(self, _filter: Dict[str,Union[int,str]] = None) -> List[TrainingContext]:
        """
        Returns a list of all available training contexts with an optional filter
        :param filter: Dictionary container the filter keyword (label, description etc) and the value to filter on (using CONTAINS)

        :return:
        """
        training_contexts = []

        sql = SQL.get_training_contexts

        if _filter:
            sql += ' WHERE '
            for k in _filter:
                sql += f"tc.{k}='{_filter[k]}' and"
            sql = sql.rstrip('and')

        training_context_rows = self.splice_ctx.df(sql)

        cols = Columns.training_context

        training_context_rows = clean_df(training_context_rows,cols)

        for tc in training_context_rows.collect():
            t = tc.asDict()
            # DB doesn't support lists so it stores , separated vals in a string
            t['pk_columns'] = t.pop('pk_columns').split(',')
            training_contexts.append(TrainingContext(**t))
        return training_contexts

    def _get_pipeline(self, df, features, label, model_type):
        categorical_features = [f.name for f in features if f.is_categorical()]
        numeric_features = [f.name for f in features if f.is_continuous() or f.is_ordinal()]
        indexed_features = [f'{n}_index' for n in categorical_features]

        si = [StringIndexer(inputCol=n, outputCol=f'{n}_index', handleInvalid='keep') for n in categorical_features]
        all_features = numeric_features + indexed_features

        v = VectorAssembler(inputCols=all_features, outputCol='features')
        if model_type=='classification':
            si += [StringIndexer(inputCol=label, outputCol=f'{label}_index', handleInvalid='keep')]
            clf = RandomForestClassifier(labelCol=f'{label}_index')
        else:
            clf = RandomForestRegressor(labelCol=label)
        return Pipeline(stages=si + [v, clf]).fit(df)

    def _get_feature_importance(self, feature_importances, df, features_column):
        feature_rank = []
        for i in df.schema[features_column].metadata["ml_attr"]["attrs"]:
            feature_rank += df.schema[features_column].metadata["ml_attr"]["attrs"][i]
        features_df = pd.DataFrame(feature_rank)
        features_df['score'] = features_df['idx'].apply(lambda x: feature_importances[x])
        return(features_df.sort_values('score', ascending = False))

    def _log_mlflow_results(self, name, rounds, mlflow_results):
        with self.mlflow_ctx.start_run(run_name=name):
            for r in range(rounds):
                with self.mlflow_ctx.start_run(run_name=f'Round {r}', nested=True):
                    for round_results in mlflow_results:
                        self.mlflow_ctx.log_metrics(round_results)

    def run_feature_elimination(self, df, features, label: str = 'label', n: int = 10, verbose: int = 0,
                            model_type: str='classification', step: int = 1, log_mlflow: bool = False,
                            mlflow_run_name: str = None, return_importances: bool = False):

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

        train_df = df
        remaining_features = features
        rnd = 0
        mlflow_results = []
        while len(remaining_features) > n:
            rnd += 1
            num_features = max(len(remaining_features)-step, n) # Don't go less than the specified value
            print(f'Building {model_type} model')
            model = self._get_pipeline(train_df, remaining_features, label, model_type)
            print('Getting feature importance')
            feature_importances = self._get_feature_importance(model.stages[-1].featureImportances, model.transform(train_df), "features").head(num_features)
            remaining_features_and_label = list(feature_importances['name'].values) + [label]
            train_df = train_df.select(*remaining_features_and_label)
            remaining_features = [f for f in remaining_features if f.name in feature_importances['name'].values]
            print(f'{len(remaining_features)} features remaining')

            if verbose == 1:
                print(f'Round {rnd} complete. Remaining Features:')
                for i,f in enumerate(list(feature_importances['name'].values)):
                    print(f'{i}. {f}')
            elif verbose == 2:
                print(f'Round {rnd} complete. Remaining Features:')
                display(feature_importances.reset_index(drop=True))

            # Add results to a list for mlflow logging
            round_metrics = {'Round':rnd, 'Number of features': remaining_features}
            for index, row in feature_importances.iterrows():
                round_metrics[row['name']] = row['score']
            mlflow_results.append(round_metrics)

        if log_mlflow and hasattr(self, 'mlflow_ctx'):
            assert hasattr(self, 'mlflow_ctx'), "In order to log results to mlflow, you must register your mlflow context!"
            run_name = mlflow_run_name or f'feature_elimination_{label}'
            self._log_mlflow_results(run_name, rnd, mlflow_results)
                        
        return remaining_features, feature_importances.reset_index(drop=True) if return_importances else remaining_features



    def get_training_context_id(self, name) -> int:
        """
        Returns the unique context ID from a name

        :param name: The training context name
        :return: The training context id
        """
        return self.splice_ctx.df(SQL.get_training_context_id.format(name=name)).collect()[0][0]

    def _validate_feature_set(self, schema, table):
        """
        Asserts a feature set doesn't already exist in the database
        :param schema: schema name of the feature set
        :param table: table name of the feature set
        :return: None
        """
        str = 'This feature set already exists. Use a different schema and/or table name.'
        # Validate Table
        assert not self.splice_ctx.tableExists(schema, table_name=table), str
        # Validate metadata
        assert len(self.get_feature_sets(_filter={'table_name': table, 'schema_name': schema})) == 0, str


    def create_feature_set(self, schema: str, table: str, primary_keys: Dict[str,str],
                           desc: Optional[str] = None) -> FeatureSet:
        """
        Creates and returns a new feature set

        :param schema:
        :param name:
        :param pk_columns:
        :param feature_column:
        :param desc:
        :return: FeatureSet
        """
        self._validate_feature_set(schema, table)
        fset = FeatureSet(splice_ctx=self.splice_ctx, schema_name=schema, table_name=table, primary_keys=primary_keys,
                            description=desc)
        # if fset in self.feature_sets:
        #     raise SpliceMachineException('This feature set already exists. Use a different schema and/or table name.')
        # self.feature_sets.append(fset)
        print(f'Registering feature set {schema}.{table} in Feature Store' )
        fset._register_metadata()
        return fset

    def get_features_by_name(self, names: List[str]) -> List[Feature]:
        """
        Returns a list of features whose names are provided
        :param names: The list of feature names
        :return: The list of features
        """
        # Format feature names into quotes strings for search
        df = self.splice_ctx.df(SQL.get_features_by_name.format(",".join([f"'{i.upper()}'" for i in names])))
        df = clean_df(df, Columns.feature)
        features = []
        for feat in df.collect():
            f = feat.asDict()
            features.append(Feature(**f))
        return features



    def create_training_context(self, *, name: str, sql: str, primary_keys: List[str], context_keys: List[str],
                            ts_col: str, label_col: Optional[str] = None, replace: Optional[bool] = False,
                            desc: Optional[str] = None) -> None:
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
        :param context_keys: (List[str]) A list of context keys in the sql that are used to get the desired features in
            get_training_set
        :param desc: (Optional[str]) An optional description of the training set
        :return:
        """

        # validate_sql()
        # register_training_context()
        label_col = f"'{label_col}'" if label_col else "NULL" # Formatting incase NULL
        train_sql = SQL.training_context.format(name=name, desc=desc or 'None Provided', sql_text=sql, ts_col=ts_col,
                                                label_col=label_col)
        print('Building training sql...')
        print('\t',train_sql)
        self.splice_ctx.execute(train_sql)
        print('Done.')

        # Get generated context ID
        cid = self.get_training_context_id(name)

        print('Creating Context Keys')
        for i in context_keys:
            key_sql = SQL.training_context_keys.format(context_id=cid, key_column=i, key_type='C')
            print(f'\tCreating Context Key {i}...')
            print('\t',key_sql)
            self.splice_ctx.execute(key_sql)
        print('Done.')
        print('Creating Primary Keys')
        for i in primary_keys:
            key_sql = SQL.training_context_keys.format(context_id=cid, key_column=i, key_type='P')
            print(f'\tCreating Primary Key {i}...')
            print('\t',key_sql)
            self.splice_ctx.execute(key_sql)
        print('Done.')


    def get_feature_context_keys(self, features: List[str]) -> Dict[str,List[str]]:
        """
        Returns a dictionary mapping each individual feature to its primary key(s)

        :param features: (List[str]) The list of features to get primary keys for
        :return: Dict[str, List[str]]
        """
        pass

    def get_available_features(self, training_context: str ) -> List[Feature]:
        """
        Returns the available features for the given a training context name

        :param training_context: The name of the training context
        :return: A list of available features
        """
        where = f"tc.Name='{training_context}'"
            
        df = self.splice_ctx.df(SQL.get_available_features.format(where=where))
        
        df = clean_df(df, Columns.feature)
        features = []
        for feat in df.collect():
            f = feat.asDict()
            features.append(Feature(**f))
        return features



    def set_feature_description(self):
        pass

    def get_feature_description(self):
        pass

    def get_training_set(self, training_context: str, features: List[Feature], start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None, return_sql: bool = False) -> SparkDF or str:
        """
        Returns the training set as a Spark Dataframe

        :param training_context: (str) The name of the registered training context
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
        cid = self.get_training_context_id(training_context)
        tctx = self.get_training_contexts(_filter={'context_id': cid})[0]
        # SELECT clause
        sql = 'SELECT '
        for pkcol in tctx.pk_columns: # Select primary key column(s)
            sql += f'\n\tctx.{pkcol},'
            cols.append(pkcol)

        sql += f'\n\tctx.{tctx.ts_column}, ' # Select timestamp column
        cols.append(tctx.ts_column)

        for feature in features:
            sql += f'\n\tCOALESCE(fset{feature.feature_set_id}.{feature.name},fset{feature.feature_set_id}h.{feature.name}) {feature.name},' # Collect all features over time
            cols.append(feature.name)

        sql = sql + f'\n\tctx.{tctx.label_column}' if tctx.label_column else sql.rstrip(',')  # Select the optional label col
        if tctx.label_column: cols.append(tctx.label_column)

        # FROM clause
        sql += f'\nFROM ({tctx.context_sql}) ctx '

        # JOIN clause
        feature_set_ids = list({f.feature_set_id for f in features}) # Distinct set of IDs
        feature_sets = self.get_feature_sets(feature_set_ids)
        for fset in feature_sets:
            # Join Feature Set
            sql += f'\nLEFT OUTER JOIN {fset.schema_name}.{fset.table_name} fset{fset.feature_set_id} \n\tON '
            for pkcol in fset.pk_columns:
                sql += f'fset{fset.feature_set_id}.{pkcol}=ctx.{pkcol} AND '
            sql += f' ctx.{tctx.ts_column} >= fset{fset.feature_set_id}.LAST_UPDATE_TS '

            # Join Feature Set History
            sql += f'\nLEFT OUTER JOIN {fset.schema_name}.{fset.table_name}_history fset{fset.feature_set_id}h \n\tON '
            for pkcol in fset.pk_columns:
                sql += f' fset{fset.feature_set_id}h.{pkcol}=ctx.{pkcol} AND '
            sql += f' ctx.{tctx.ts_column} >= fset{fset.feature_set_id}h.ASOF_TS AND ctx.{tctx.ts_column} < fset{fset.feature_set_id}h.UNTIL_TS'

        # WHERE clause on optional start and end times
        if start_time or end_time:
            sql += '\nWHERE '
            if start_time:
                sql += f"\n\tctx.{tctx.ts_column} >= '{str(start_time)}' AND"
            if end_time:
                sql += f"\n\tctx.{tctx.ts_column} <= '{str(end_time)}'"
            sql = sql.rstrip('AND')

        return sql if return_sql else clean_df(self.splice_ctx.df(sql),cols)


    def get_feature_vector_sql(self, training_context: str, features: List[Feature], include_insert: Optional[bool] = True  ) -> str:
        """
        Returns the parameterized feature retrieval SQL used for online model serving.

        :param training_context_id: (str) The name of the registered training context
        :param features: (List[str]) the list of features from the feature store to be included in the training
            * NOTE: This function will error if the context SQL is missing a context key required to retrieve the\
             desired features
        :param include_insert: (Optional[bool]) determines whether insert into model table is included in the SQL statement
        :return : (str)
        """

        # Get training context information (ctx primary key column(s), ctx primary key inference ts column, )
        cid = self.get_training_context_id(training_context)
        tctx = self.get_training_contexts(_filter={'context_id': cid})[0]

        # optional INSERT prefix
        if (include_insert):
            sql = 'INSERT INTO {target_model_table} ('
            for pkcol in tctx.pk_columns: # Select primary key column(s)
                sql += f'{pkcol}, '
            for feature in features:
                sql += f'{feature.name}, ' # Collect all features over time
            sql = sql.rstrip(', ')
            sql += ')\nSELECT '
        else:
            sql = 'SELECT '

        # SELECT expressions
        for pkcol in tctx.pk_columns: # Select primary key column(s)
            sql += f'\n\t{{p_{pkcol}}} {pkcol},'

        for feature in features:
            sql += f'\n\tfset{feature.feature_set_id}.{feature.name}, ' # Collect all features over time
        sql = sql.rstrip(', ')

        # FROM clause
        sql += f'\nFROM '

        # JOIN clause
        feature_set_ids = list({f.feature_set_id for f in features}) # Distinct set of IDs
        feature_sets = self.get_feature_sets(feature_set_ids)
        where = '\nWHERE '
        for fset in feature_sets:
            # Join Feature Set
            sql += f'\n\t{fset.schema_name}.{fset.table_name} fset{fset.feature_set_id}, '
            for pkcol in fset.pk_columns:
                where += f'\n\tfset{fset.feature_set_id}.{pkcol}={{p_{pkcol}}} AND '

        sql = sql.rstrip(', ')
        where = where.rstrip('AND ')
        sql += where

        return sql


    def list_training_sets(self) -> Dict[str, Optional[str]]:
        """
        Returns a dictionary a training sets available, with the map name -> description. If there is no description,
        the value will be an emtpy string

        :return: Dict[str, Optional[str]]
        """
        pass
