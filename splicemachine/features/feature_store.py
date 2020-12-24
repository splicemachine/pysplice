from typing import List, Dict, Optional, Union
from datetime import datetime
import re

from IPython.display import display
import pandas as pd

from pyspark.sql.dataframe import DataFrame as SparkDF
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import StringIndexer, VectorAssembler

from splicemachine import SpliceMachineException
from splicemachine.spark import PySpliceContext
from splicemachine.features import Feature, FeatureSet
from .training_set import TrainingSet

from .constants import SQL, Columns, FeatureType
from .training_context import TrainingContext
from .utils import clean_df

class FeatureStore:
    def __init__(self, splice_ctx: PySpliceContext) -> None:
        self.splice_ctx = splice_ctx
        self.feature_sets = []  # Cache of newly created feature sets

    def register_splice_context(self, splice_ctx: PySpliceContext) -> None:
        self.splice_ctx = splice_ctx

    def get_feature_sets(self, feature_set_ids: List[int] = None, _filter: Dict[str, str] = None) -> List[FeatureSet]:
        """
        Returns a list of available feature sets

        :param feature_set_ids: A list of feature set IDs. If none will return all FeatureSets
        :param _filter: Dictionary of filters to apply to the query. This filter can be on any attribute of FeatureSets.
            If None, will return all FeatureSets
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
            d['primary_keys'] = {c: k for c, k in zip(pkcols, pktypes)}
            feature_sets.append(FeatureSet(splice_ctx=self.splice_ctx, **d))
        return feature_sets

    def get_feature_dataset(self, features: Union[List[Feature], List[str]]):
        """
        Gets a set of feature values across feature sets that is not time dependent (ie for non time series clustering)
        :param features: List of Features or strings of feature names
        :return: Spark DF
        """
        features = self.get_features_by_name(names=features, as_list=True) \
            if all([isinstance(i,str) for i in features]) else features

        sql = SQL.get_feature_set_join_keys.format(names=tuple([f.name for f in features]))
        fset_keys: pd.DataFrame = self.splice_ctx.df(sql).toPandas()
        # Get max number of pk (join) columns from all feature sets
        fset_keys['PK_COLUMNS_COUNT'] = fset_keys['PK_COLUMNS'].apply(lambda x: len(x.split('|')))
        # Get "anchor" feature set. The one we will use to try to join to all others
        ind = fset_keys['PK_COLUMNS_COUNT'].idxmax()
        anchor_series = fset_keys.iloc[ind]
        # Remove that from the list
        fset_keys.drop(index=ind, inplace=True)
        all_pk_cols = anchor_series.PK_COLUMNS.split('|')
        # For each feature set, assert that all join keys exist in our "anchor" feature set
        fset_keys['can_join'] = fset_keys['PK_COLUMNS'].map(lambda x: set(x.split('|')).issubset(all_pk_cols) )
        if not fset_keys['can_join'].all():
            bad_feature_set_ids = [t.FEATURE_SET_ID for _ , t in fset_keys[fset_keys['can_join'] != True].iterrows()]
            bad_features = [f.name for f in features if f.feature_set_id in bad_feature_set_ids]
            raise SpliceMachineException(f"The provided features do not have a common join key."
                                         f"Remove features {bad_features} from your request")

        # SELECT clause
        sql = 'SELECT '

        sql += ','.join([f'fset{feature.feature_set_id}.{feature.name}' for feature in features])

        alias = f'fset{anchor_series.FEATURE_SET_ID}' # We use this a lot for joins
        sql += f'\nFROM {anchor_series.SCHEMA_NAME}.{anchor_series.TABLE_NAME} {alias} '

        # JOIN clause
        for _, fset in fset_keys.iterrows():
            # Join Feature Set
            sql += f'\nLEFT OUTER JOIN {fset.SCHEMA_NAME}.{fset.TABLE_NAME} fset{fset.FEATURE_SET_ID} \n\tON '
            for ind, pkcol in enumerate(fset.PK_COLUMNS.split('|')):
                if ind > 0: sql += ' AND ' # In case of multiple columns
                sql += f'fset{fset.FEATURE_SET_ID}.{pkcol}={alias}.{pkcol}'

        # Link this to mlflow for model deployment
        # Here we create a null training context and pass it into the training set. We do this because this special kind
        # of training set isn't standard. It's not based on a training context, on primary key columns, a label column,
        # or a timestamp column . This is simply a joined set of features from different feature sets.
        # But we still want to track this in mlflow as a user may build and deploy a model based on this. So we pass in
        # a null training context that can be tracked with a "name" (although the name is None). This is a likely case
        # for (non time based) clustering use cases.
        null_tx = TrainingContext(pk_columns=[], ts_column=None, label_column=None, context_sql=None, name=None,
                                  description=None)
        ts = TrainingSet(training_context=null_tx, features=features)
        if hasattr(self, 'mlflow_ctx'):
            self.mlflow_ctx._active_training_set: TrainingSet = ts
            ts._register_metadata(self.mlflow_ctx)
        return self.splice_ctx.df(sql)


    def remove_training_context(self, override=False):
        """
        Removes a training context. This will run 2 checks.
        1. See if the training context is being used by a model in a deployment. If this is the case, the function will fail, always.
        2. See if the training context is being used in any mlflow runs (non-deployed models). This will fail and return
        a warning Telling the user that this training context is being used in mlflow runs (and the run_ids) and that
        they will need to "override" this function to forcefully remove the training context.
        """
        pass

    def get_training_context(self, training_context: str) -> TrainingContext:
        """
        Gets a training context by name

        :param training_context: Training context name
        :return: TrainingContext
        """
        return self.get_training_contexts(_filter={'name': training_context})[0]

    def get_training_contexts(self, _filter: Dict[str, Union[int, str]] = None) -> List[TrainingContext]:
        """
        Returns a list of all available training contexts with an optional filter

        :param _filter: Dictionary container the filter keyword (label, description etc) and the value to filter on
            If None, will return all TrainingContexts
        :return: List[TrainingContext]
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

        training_context_rows = clean_df(training_context_rows, cols)

        for tc in training_context_rows.collect():
            t = tc.asDict()
            # DB doesn't support lists so it stores , separated vals in a string
            t['pk_columns'] = t.pop('pk_columns').split(',')
            training_contexts.append(TrainingContext(**t))
        return training_contexts

    def get_training_context_id(self, name: str) -> int:
        """
        Returns the unique context ID from a name

        :param name: The training context name
        :return: The training context id
        """
        return self.splice_ctx.df(SQL.get_training_context_id.format(name=name)).collect()[0][0]

    def get_features_by_name(self, names: Optional[List[str]] = None, as_list=False) -> Union[List[Feature], SparkDF]:
        """
        Returns a dataframe or list of features whose names are provided

        :param names: The list of feature names
        :param as_list: Whether or not to return a list of features. Default False
        :return: The list of features or Spark Dataframe
        """
        # If they don't pass in feature names, get all features
        where_clause = "name in (" + ",".join([f"'{i.upper()}'" for i in names]) + ")" if names else "1=1"
        df = self.splice_ctx.df(SQL.get_features_by_name.format(where=where_clause))
        if not as_list: return df

        features = []
        for feat in df.collect():
            f = feat.asDict()
            f = dict((k.lower(), v) for k, v in f.items()) # DB returns uppercase column names
            features.append(Feature(**f))
        return features

    def remove_feature_set(self):
        # TODO
        raise NotImplementedError

    def get_feature_vector_sql(self, training_context: str, features: List[Feature],
                               include_insert: Optional[bool] = True) -> str:
        """
        Returns the parameterized feature retrieval SQL used for online model serving.

        :param training_context: (str) The name of the registered training context
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
            for pkcol in tctx.pk_columns:  # Select primary key column(s)
                sql += f'{pkcol}, '
            for feature in features:
                sql += f'{feature.name}, '  # Collect all features over time
            sql = sql.rstrip(', ')
            sql += ')\nSELECT '
        else:
            sql = 'SELECT '

        # SELECT expressions
        for pkcol in tctx.pk_columns:  # Select primary key column(s)
            sql += f'\n\t{{p_{pkcol}}} {pkcol},'

        for feature in features:
            sql += f'\n\tfset{feature.feature_set_id}.{feature.name}, '  # Collect all features over time
        sql = sql.rstrip(', ')

        # FROM clause
        sql += f'\nFROM '

        # JOIN clause
        feature_set_ids = list({f.feature_set_id for f in features})  # Distinct set of IDs
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

    def get_feature_context_keys(self, features: List[str]) -> Dict[str, List[str]]:
        """
        Returns a dictionary mapping each individual feature to its primary key(s)

        :param features: (List[str]) The list of features to get primary keys for
        :return: Dict[str, List[str]]
        """
        pass

    def get_training_context_features(self, training_context: str) -> List[Feature]:
        """
        Returns the available features for the given a training context name

        :param training_context: The name of the training context
        :return: A list of available features
        """
        where = f"tc.Name='{training_context}'"

        df = self.splice_ctx.df(SQL.get_training_context_features.format(where=where))

        df = clean_df(df, Columns.feature)
        features = []
        for feat in df.collect():
            f = feat.asDict()
            features.append(Feature(**f))
        return features

    def get_feature_description(self):
        #TODO
        raise NotImplementedError

    def get_training_set(self, training_context: str, features: Union[List[Feature],List[str]], start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None, return_sql: bool = False) -> SparkDF or str:
        """
        Returns the training set as a Spark Dataframe

        :param training_context: (str) The name of the registered training context
        :param features: (List[str] OR List[Feature]) the list of features from the feature store to be included in the training.
            If a list of strings is passed in it will be converted to a list of Feature
            * NOTE: This function will error if the context SQL is missing a context key required to retrieve the\
             desired features
        :param start_time: (Optional[datetime]) The start time of the query (how far back in the data to start). Default None
            * NOTE: If start_time is None, query will start from beginning of history
        :param end_time: (Optional[datetime]) The end time of the query (how far recent in the data to get). Default None
            * NOTE: If end_time is None, query will get most recently available data
        :param return_sql: (Optional[bool]) Return the SQL statement (str) instead of the Spark DF. Defaults False
        :return: Optional[SparkDF, str]
        """

        features = self.get_features_by_name(names=features, as_list=True) if all([isinstance(i,str) for i in features]) else features
        # DB-9556 loss of column names on complex sql for NSDS
        cols = []

        # Get training context information (ctx primary key column(s), ctx primary key inference ts column, )
        tctx = self.get_training_context(training_context)
        # SELECT clause
        sql = 'SELECT '
        for pkcol in tctx.pk_columns:  # Select primary key column(s)
            sql += f'\n\tctx.{pkcol},'
            cols.append(pkcol)

        sql += f'\n\tctx.{tctx.ts_column}, '  # Select timestamp column
        cols.append(tctx.ts_column)

        # TODO: ensure these features exist and fail gracefully if not
        for feature in features:
            sql += f'\n\tCOALESCE(fset{feature.feature_set_id}.{feature.name},fset{feature.feature_set_id}h.{feature.name}) {feature.name},'  # Collect all features over time
            cols.append(feature.name)

        sql = sql + f'\n\tctx.{tctx.label_column}' if tctx.label_column else sql.rstrip(
            ',')  # Select the optional label col
        if tctx.label_column: cols.append(tctx.label_column)

        # FROM clause
        sql += f'\nFROM ({tctx.context_sql}) ctx '

        # JOIN clause
        feature_set_ids = list({f.feature_set_id for f in features})  # Distinct set of IDs
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

        # Link this to mlflow for model deployment
        if hasattr(self, 'mlflow_ctx'):
            ts = TrainingSet(training_context=tctx, features=features,
                                                               start_time=start_time, end_time=end_time)
            self.mlflow_ctx._active_training_set: TrainingSet = ts
            ts._register_metadata(self.mlflow_ctx)

        return sql if return_sql else clean_df(self.splice_ctx.df(sql), cols)

    def list_training_sets(self) -> Dict[str, Optional[str]]:
        """
        Returns a dictionary a training sets available, with the map name -> description. If there is no description,
        the value will be an emtpy string

        :return: Dict[str, Optional[str]]
        """
        raise NotImplementedError("To see available training contexts, run fs.describe_training_contexts()")

    def _validate_feature_set(self, schema_name, table_name):
        """
        Asserts a feature set doesn't already exist in the database
        :param schema_name: schema name of the feature set
        :param table_name: table name of the feature set
        :return: None
        """
        str = f'Feature Set {schema_name}.{table_name} already exists. Use a different schema and/or table name.'
        # Validate Table
        assert not self.splice_ctx.tableExists(schema_name, table_name=table_name), str
        # Validate metadata
        assert len(self.get_feature_sets(_filter={'table_name': table_name, 'schema_name': schema_name})) == 0, str

    def create_feature_set(self, schema_name: str, table_name: str, primary_keys: Dict[str, str],
                           desc: Optional[str] = None) -> FeatureSet:
        """
        Creates and returns a new feature set

        :param schema_name: The schema under which to create the feature set table
        :param table_name: The table name for this feature set
        :param primary_keys: The primary key column(s) of this feature set
        :param desc: The (optional) description
        :return: FeatureSet
        """
        self._validate_feature_set(schema_name, table_name)
        fset = FeatureSet(splice_ctx=self.splice_ctx, schema_name=schema_name, table_name=table_name, primary_keys=primary_keys,
                          description=desc)
        self.feature_sets.append(fset)
        print(f'Registering feature set {schema_name}.{table_name} in Feature Store')
        fset._register_metadata()
        return fset

    def _validate_feature(self, name):
        """
        Ensures that the feature doesn't exist as all features have unique names
        :param name: the Feature name
        :return:
        """
        # TODO: Capitalization of feature name column
        # TODO: Make more informative, add which feature set contains existing feature
        str = f"Cannot add feature {name}, feature already exists in Feature Store. Try a new feature name."
        l = self.splice_ctx.df(SQL.get_all_features.format(name=name.upper())).count()
        assert l == 0, str

        if not re.match('^[A-Za-z][A-Za-z0-9_]*$', name):
            raise SpliceMachineException('Feature name does not conform. Must start with an alphabetic character, '
                                         'and can only contains letters, numbers and underscores')

    def __validate_feature_data_type(self, feature_data_type: str):
        """
        Validated that the provided feature data type is a valid SQL data type
        :param feature_data_type: the feature data type
        :return: None
        """
        from splicemachine.spark.constants import SQL_TYPES
        if not feature_data_type.split('(')[0] in SQL_TYPES:
            raise SpliceMachineException(f"The datatype you've passed in, {feature_data_type} is not a valid SQL type. "
                                         f"Valid types are {SQL_TYPES}")

    def create_feature(self, schema_name: str, table_name: str, name: str, feature_data_type: str,
                       feature_type: FeatureType, desc: str = None, tags: List[str] = None):
        """
        Add a feature to a feature set

        :param schema_name: The feature set schema
        :param table_name: The feature set table name to add the feature to
        :param name: The feature name
        :param feature_data_type: The datatype of the feature. Must be a valid SQL datatype
        :param feature_type: splicemachine.features.FeatureType of the feature. Available are FeatureType.[categorical, ordinal, continuous]
        :param desc: The (optional) feature description (default None)
        :param tags: (optional) List of (str) tag words (default None)
        :return:
        """
        self.__validate_feature_data_type(feature_data_type)
        if self.splice_ctx.tableExists(schema_name, table_name):
            raise SpliceMachineException(f"Feature Set {schema_name}.{table_name} is already deployed. You cannot "
                                         f"add features to a deployed feature set.")
        fset: FeatureSet = self.get_feature_sets(_filter={'table_name': table_name, 'schema_name': schema_name})[0]
        self._validate_feature(name)
        f = Feature(name=name, description=desc or '', feature_data_type=feature_data_type,
                    feature_type=feature_type, tags=tags or [], feature_set_id=fset.feature_set_id)
        print(f'Registering feature {f.name} in Feature Store')
        f._register_metadata(self.splice_ctx)
        # TODO: Backfill the feature

    def _validate_training_context(self, name, sql, context_keys):
        """
        Validates that the training context doesn't already exist.

        :param name: The training context name
        :param sql: The training context provided SQL
        :param context_keys: The provided context keys when creating the training context
        :return:
        """
        # Validate name doesn't exist
        assert len(self.get_training_contexts(_filter={'name': name})) == 0, f"Training context {name} already exists!"

        # Column comparison
        # Lazily evaluate sql resultset, ensure that the result contains all columns matching pks, context_keys, tscol and label_col
        from py4j.protocol import Py4JJavaError
        try:
            context_sql_df = self.splice_ctx.df(sql)
        except Py4JJavaError as e:
            if 'SQLSyntaxErrorException' in str(e.java_exception):
                raise SpliceMachineException(f'The provided SQL is incorrect. The following error was raised during '
                                             f'validation:\n\n{str(e.java_exception)}') from None
            raise e

        # Confirm that all context_keys provided correspond to primary keys of created feature sets
        pks = set(i[0].upper() for i in self.splice_ctx.df(SQL.get_fset_primary_keys).collect())
        missing_keys = set(i.upper() for i in context_keys) - pks
        assert not missing_keys, f"Not all provided context keys exist. Remove {missing_keys} or " \
                                 f"create a feature set that uses the missing keys"

    def create_training_context(self, name: str, sql: str, primary_keys: List[str], context_keys: List[str],
                                ts_col: str, label_col: Optional[str] = None, replace: Optional[bool] = False,
                                desc: Optional[str] = None, verbose=False) -> None:
        """
        Registers a training context for use in generating training SQL

        :param name: The training set name. This must be unique to other existing training sets unless replace is True
        :param sql: (str) a SELECT statement that includes:
            * the primary key column(s) - uniquely identifying a training row/case
            * the inference timestamp column - timestamp column with which to join features (temporal join timestamp)
            * context key(s) - the references to the other feature tables' primary keys (ie customer_id, location_id)
            * (optionally) the label expression - defining what the training set is trying to predict
        :param primary_keys: (List[str]) The list of columns from the training SQL that identify the training row
        :param ts_col: The timestamp column of the training SQL that identifies the inference timestamp
        :param label_col: (Optional[str]) The optional label column from the training SQL.
        :param replace: (Optional[bool]) Whether to replace an existing training context
        :param context_keys: (List[str]) A list of context keys in the sql that are used to get the desired features in
            get_training_set
        :param desc: (Optional[str]) An optional description of the training set
        :param verbose: Whether or not to print the SQL before execution (default False)
        :return:
        """
        assert name != "None", "Name of training context cannot be None!"
        self._validate_training_context(name, sql, context_keys)
        # register_training_context()
        label_col = f"'{label_col}'" if label_col else "NULL"  # Formatting incase NULL
        train_sql = SQL.training_context.format(name=name, desc=desc or 'None Provided', sql_text=sql, ts_col=ts_col,
                                                label_col=label_col)
        print('Building training sql...')
        if verbose: print('\t', train_sql)
        self.splice_ctx.execute(train_sql)
        print('Done.')

        # Get generated context ID
        cid = self.get_training_context_id(name)

        print('Creating Context Keys')
        for i in context_keys:
            key_sql = SQL.training_context_keys.format(context_id=cid, key_column=i.upper(), key_type='C')
            print(f'\tCreating Context Key {i}...')
            if verbose: print('\t', key_sql)
            self.splice_ctx.execute(key_sql)
        print('Done.')
        print('Creating Primary Keys')
        for i in primary_keys:
            key_sql = SQL.training_context_keys.format(context_id=cid, key_column=i.upper(), key_type='P')
            print(f'\tCreating Primary Key {i}...')
            if verbose: print('\t', key_sql)
            self.splice_ctx.execute(key_sql)
        print('Done.')

    def deploy_feature_set(self, schema_name, table_name):
        """
        Deploys a feature set to the database. This persists the feature stores existence.
        As of now, once deployed you cannot delete the feature set or add/delete features.
        The feature set must have already been created with create_feature_set
        :param schema_name: The schema of the created feature set
        :param table_name: The table of the created feature set
        """
        try:
            fset = self.get_feature_sets(_filter={'schema_name': schema_name, 'table_name': table_name})[0]
        except:
            raise SpliceMachineException(f"Cannot find feature set {schema_name}.{table_name}. Ensure you've created this"
                                         f"feature set using fs.create_feature_set before deploying.")
        fset.deploy()

    def describe_feature_sets(self) -> None:
        """
        Prints out a description of a all feature sets, with all features in the feature sets and whether the feature
        set is deployed

        :return: None
        """
        print('Available feature sets')
        for fset in self.get_feature_sets():
            print('-' * 200)
            self.describe_feature_set(fset.schema_name, fset.table_name)

    def describe_feature_set(self, schema_name: str, table_name: str) -> None:
        """
        Prints out a description of a given feature set, with all features in the feature set and whether the feature
        set is deployed

        :param schema_name: feature set schema name
        :param table_name: feature set table name
        :return: None
        """
        fset = self.get_feature_sets(_filter={'schema_name': schema_name, 'table_name': table_name})
        if not fset: raise SpliceMachineException(f"Feature Set {schema_name}.{table_name} not found. Check name and try again.")
        fset = fset[0]
        print(f'{fset.schema_name}.{fset.table_name} - {fset.description}')
        print('Primary keys:', fset.primary_keys)
        print('\nAvailable features:')
        display(pd.DataFrame(f.__dict__ for f in fset.get_features()))

    def describe_training_contexts(self) -> None:
        """
        Prints out a description of all training contexts, the ID, name, description and optional label

        :param training_context: The training context name
        :return: None
        """
        print('Available training contexts')
        for tcx in self.get_training_contexts():
            print('-' * 200)
            self.describe_training_context(tcx.name)

    def describe_training_context(self, training_context: str) -> None:
        """
        Prints out a description of a given training context, the ID, name, description and optional label

        :param training_context: The training context name
        :return: None
        """
        tcx = self.get_training_contexts(_filter={'name': training_context})
        if not tcx: raise SpliceMachineException(f"Training context {training_context} not found. Check name and try again.")
        tcx = tcx[0]
        print(f'ID({tcx.context_id}) {tcx.name} - {tcx.description} - LABEL: {tcx.label_column}')
        print(f'Available features in {tcx.name}:')
        feats: List[Feature] = self.get_training_context_features(tcx.name)
        # Grab the feature set info and their corresponding names (schema.table) for the display table
        feat_sets: List[FeatureSet] = self.get_feature_sets(feature_set_ids=[f.feature_set_id for f in feats])
        feat_sets: Dict[int,str] = {fset.feature_set_id: f'{fset.schema_name}.{fset.table_name}' for fset in feat_sets}
        for f in feats:
            f.feature_set_name = feat_sets[f.feature_set_id]
        display(pd.DataFrame(f.__dict__ for f in feats))

    def set_feature_description(self):
        raise NotImplementedError

    def __get_pipeline(self, df, features, label, model_type):
        """
        Creates a Pipeline with preprocessing steps (StringINdexer, VectorAssembler) for each feature depending
        on feature type, and returns the pipeline for training for feature elimination

        :param df: Spark Dataframe
        :param features: List[Feature] to train on
        :param label: Label name to train on
        :param model_type: (str) the model type - avl options are "classification" and "regression"
        :return: Unfit Spark Pipeline
        """
        categorical_features = [f.name for f in features if f.is_categorical()]
        numeric_features = [f.name for f in features if f.is_continuous() or f.is_ordinal()]
        indexed_features = [f'{n}_index' for n in categorical_features]

        si = [StringIndexer(inputCol=n, outputCol=f'{n}_index', handleInvalid='keep') for n in categorical_features]
        all_features = numeric_features + indexed_features

        v = VectorAssembler(inputCols=all_features, outputCol='features')
        if model_type == 'classification':
            si += [StringIndexer(inputCol=label, outputCol=f'{label}_index', handleInvalid='keep')]
            clf = RandomForestClassifier(labelCol=f'{label}_index')
        else:
            clf = RandomForestRegressor(labelCol=label)
        return Pipeline(stages=si + [v, clf]).fit(df)

    def __get_feature_importance(self, feature_importances, df, features_column):
        """
        Gets the ordered feature importance for the feature elimination rounds
        :param feature_importances: Spark model featureImportances attribute
        :param df: Spark dataframe
        :param features_column: Column name of the dataframe that holds the features
        :return: Sorted pandas dataframe with the feature importances and feature names
        """
        feature_rank = []
        for i in df.schema[features_column].metadata["ml_attr"]["attrs"]:
            feature_rank += df.schema[features_column].metadata["ml_attr"]["attrs"][i]
        features_df = pd.DataFrame(feature_rank)
        features_df['score'] = features_df['idx'].apply(lambda x: feature_importances[x])
        return (features_df.sort_values('score', ascending=False))

    def __log_mlflow_results(self, name, rounds, mlflow_results):
        """
        Logs the results of feature elimination to mlflow
        :param name: MLflow run name
        :param rounds: Number of rounds of feature elimination that were run
        :param mlflow_results: The params / metrics to log
        :return:
        """
        with self.mlflow_ctx.start_run(run_name=name):
            for r in range(rounds):
                with self.mlflow_ctx.start_run(run_name=f'Round {r}', nested=True):
                    self.mlflow_ctx.log_metrics(mlflow_results[r])

    def run_feature_elimination(self, df, features, label: str = 'label', n: int = 10, verbose: int = 0,
                                model_type: str = 'classification', step: int = 1, log_mlflow: bool = False,
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
        assert len(
            remaining_features) > n, \
            "You've set the number of desired features (n) greater than the number of available features"
        while len(remaining_features) > n:
            rnd += 1
            num_features = max(len(remaining_features) - step, n)  # Don't go less than the specified value
            print(f'Building {model_type} model')
            model = self.__get_pipeline(train_df, remaining_features, label, model_type)
            print('Getting feature importance')
            feature_importances = self.__get_feature_importance(model.stages[-1].featureImportances,
                                                                model.transform(train_df), "features").head(num_features)
            remaining_features_and_label = list(feature_importances['name'].values) + [label]
            train_df = train_df.select(*remaining_features_and_label)
            remaining_features = [f for f in remaining_features if f.name in feature_importances['name'].values]
            print(f'{len(remaining_features)} features remaining')

            if verbose == 1:
                print(f'Round {rnd} complete. Remaining Features:')
                for i, f in enumerate(list(feature_importances['name'].values)):
                    print(f'{i}. {f}')
            elif verbose == 2:
                print(f'Round {rnd} complete. Remaining Features:')
                display(feature_importances.reset_index(drop=True))

            # Add results to a list for mlflow logging
            round_metrics = {'Round': rnd, 'Number of features': len(remaining_features)}
            for index, row in feature_importances.iterrows():
                round_metrics[row['name']] = row['score']
            mlflow_results.append(round_metrics)

        if log_mlflow and hasattr(self, 'mlflow_ctx'):
            run_name = mlflow_run_name or f'feature_elimination_{label}'
            self.__log_mlflow_results(run_name, rnd, mlflow_results)

        return remaining_features, feature_importances.reset_index(
            drop=True) if return_importances else remaining_features
