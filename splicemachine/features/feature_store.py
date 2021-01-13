from typing import List, Dict, Optional, Union
from datetime import datetime
import re

from IPython.display import display
import pandas as pd
from pandas import DataFrame as PandasDF

from pyspark.sql.dataframe import DataFrame as SparkDF
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import StringIndexer, VectorAssembler

from splicemachine import SpliceMachineException
from splicemachine.spark import PySpliceContext
from splicemachine.features import Feature, FeatureSet
from .training_set import TrainingSet
from .utils.training_utils import (dict_to_lower, _generate_training_set_history_sql,
                                   _generate_training_set_sql, _create_temp_training_view)
from .constants import SQL, FeatureType
from .training_view import TrainingView


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
        :return: List[FeatureSet] the list of Feature Sets
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

        feature_set_rows = self.splice_ctx.df(sql, to_lower=True)

        for fs in feature_set_rows.collect():
            d = fs.asDict()
            pkcols = d.pop('pk_columns').split('|')
            pktypes = d.pop('pk_types').split('|')
            d['primary_keys'] = {c: k for c, k in zip(pkcols, pktypes)}
            feature_sets.append(FeatureSet(splice_ctx=self.splice_ctx, **d))
        return feature_sets

    def remove_training_view(self, override=False):
        """
        Note: This function is not yet implemented.
        Removes a training view. This will run 2 checks.
            1. See if the training view is being used by a model in a deployment. If this is the case, the function will fail, always.
            2. See if the training view is being used in any mlflow runs (non-deployed models). This will fail and return
            a warning Telling the user that this training view is being used in mlflow runs (and the run_ids) and that
            they will need to "override" this function to forcefully remove the training view.
        """
        raise NotImplementedError

    def get_training_view(self, training_view: str) -> TrainingView:
        """
        Gets a training view by name

        :param training_view: Training view name
        :return: TrainingView
        """
        return self.get_training_views(_filter={'name': training_view})[0]

    def get_training_views(self, _filter: Dict[str, Union[int, str]] = None) -> List[TrainingView]:
        """
        Returns a list of all available training views with an optional filter

        :param _filter: Dictionary container the filter keyword (label, description etc) and the value to filter on
            If None, will return all TrainingViews
        :return: List[TrainingView]
        """
        training_views = []

        sql = SQL.get_training_views

        if _filter:
            sql += ' WHERE '
            for k in _filter:
                sql += f"tc.{k}='{_filter[k]}' and"
            sql = sql.rstrip('and')

        training_view_rows = self.splice_ctx.df(sql, to_lower=True)

        for tc in training_view_rows.collect():
            t = tc.asDict()
            # DB doesn't support lists so it stores , separated vals in a string
            t['pk_columns'] = t.pop('pk_columns').split(',')
            training_views.append(TrainingView(**t))
        return training_views

    def get_training_view_id(self, name: str) -> int:
        """
        Returns the unique view ID from a name

        :param name: The training view name
        :return: The training view id
        """
        return self.splice_ctx.df(SQL.get_training_view_id.format(name=name)).collect()[0][0]

    def get_features_by_name(self, names: Optional[List[str]], as_list=False) -> Union[List[Feature], SparkDF]:
        """
        Returns a dataframe or list of features whose names are provided

        :param names: The list of feature names
        :param as_list: Whether or not to return a list of features. Default False
        :return: SparkDF or List[Feature] The list of Feature objects or Spark Dataframe of features and their metadata. Note, this is not the Feature
        values, simply the describing metadata about the features. To create a training dataset with Feature values, see
        :py:meth:`features.FeatureStore.get_training_set` or :py:meth:`features.FeatureStore.get_feature_dataset`
        """
        # If they don't pass in feature names, get all features
        where_clause = "name in (" + ",".join([f"'{i.upper()}'" for i in names]) + ")"
        df = self.splice_ctx.df(SQL.get_features_by_name.format(where=where_clause), to_lower=True)
        if not as_list: return df

        features = []
        for feat in df.collect():
            f = feat.asDict()
            f = dict((k.lower(), v) for k, v in f.items())  # DB returns uppercase column names
            features.append(Feature(**f))
        return features

    def remove_feature_set(self):
        # TODO
        raise NotImplementedError

    def _validate_feature_vector_keys(self, join_key_values, feature_sets) -> None:
        """
        Validates that all necessary primary keys are provided when requesting a feature vector

        :param join_key_values: dict The primary (join) key columns and values provided by the user
        :param feature_sets: List[FeatureSet] the list of Feature Sets derived from the requested Features
        :return: None. Raise Exception on bad validation
        """

        feature_set_key_columns = {fkey.lower() for fset in feature_sets for fkey in fset.primary_keys.keys()}
        missing_keys = feature_set_key_columns - join_key_values.keys()
        assert not missing_keys, f"The following keys were not provided and must be: {missing_keys}"

    def get_feature_vector(self, features: List[Union[str, Feature]],
                           join_key_values: Dict[str, str], return_sql=False) -> Union[str, PandasDF]:
        """
        Gets a feature vector given a list of Features and primary key values for their corresponding Feature Sets

        :param features: List of str Feature names or Features
        :param join_key_values: (dict) join key vals to get the proper Feature values formatted as {join_key_column_name: join_key_value}
        :param return_sql: Whether to return the SQL needed to get the vector or the values themselves. Default False
        :return: Pandas Dataframe or str (SQL statement)
        """
        feats: List[Feature] = self._process_features(features)
        # Match the case of the keys
        join_keys = dict_to_lower(join_key_values)

        # Get the feature sets and their primary key column names
        feature_sets = self.get_feature_sets([f.feature_set_id for f in feats])
        self._validate_feature_vector_keys(join_keys, feature_sets)

        feature_names = ','.join([f.name for f in feats])
        fset_tables = ','.join(
            [f'{fset.schema_name}.{fset.table_name} fset{fset.feature_set_id}' for fset in feature_sets])
        sql = "SELECT {feature_names} FROM {fset_tables} ".format(feature_names=feature_names, fset_tables=fset_tables)

        # For each Feature Set, for each primary key in the given feature set, get primary key value from the user provided dictionary
        pk_conditions = [f"fset{fset.feature_set_id}.{pk_col} = {join_keys[pk_col.lower()]}"
                         for fset in feature_sets for pk_col in fset.primary_keys]
        pk_conditions = ' AND '.join(pk_conditions)

        sql += f"WHERE {pk_conditions}"

        return sql if return_sql else self.splice_ctx.df(sql).toPandas()

    def get_feature_vector_sql_from_training_view(self, training_view: str, features: List[Feature]) -> str:
        """
        Returns the parameterized feature retrieval SQL used for online model serving.

        :param training_view: (str) The name of the registered training view
        :param features: (List[str]) the list of features from the feature store to be included in the training

            :NOTE:
                .. code-block:: text

                    This function will error if the view SQL is missing a view key required to retrieve the\
                    desired features

        :return: (str) the parameterized feature vector SQL
        """

        # Get training view information (ctx primary key column(s), ctx primary key inference ts column, )
        vid = self.get_training_view_id(training_view)
        tctx = self.get_training_views(_filter={'view_id': vid})[0]

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

    def get_feature_primary_keys(self, features: List[str]) -> Dict[str, List[str]]:
        """
        Returns a dictionary mapping each individual feature to its primary key(s). This function is not yet implemented.

        :param features: (List[str]) The list of features to get primary keys for
        :return: Dict[str, List[str]] A mapping of {feature name: [pk1, pk2, etc]}
        """
        pass

    def get_training_view_features(self, training_view: str) -> List[Feature]:
        """
        Returns the available features for the given a training view name

        :param training_view: The name of the training view
        :return: A list of available Feature objects
        """
        where = f"tc.Name='{training_view}'"

        df = self.splice_ctx.df(SQL.get_training_view_features.format(where=where), to_lower=True)

        features = []
        for feat in df.collect():
            f = feat.asDict()
            features.append(Feature(**f))
        return features

    def get_feature_description(self):
        # TODO
        raise NotImplementedError

    def get_training_set(self, features: Union[List[Feature], List[str]], current_values_only: bool = False,
                         start_time: datetime = None, end_time: datetime = None, return_sql: bool = False) -> SparkDF:
        """
        Gets a set of feature values across feature sets that is not time dependent (ie for non time series clustering).
        This feature dataset will be treated and tracked implicitly the same way a training_dataset is tracked from
        :py:meth:`features.FeatureStore.get_training_set` . The dataset's metadata and features used will be tracked in mlflow automatically (see
        get_training_set for more details).

        The way point-in-time correctness is guaranteed here is by choosing one of the Feature Sets as the "anchor" dataset.
        This means that the points in time that the query is based off of will be the points in time in which the anchor
        Feature Set recorded changes. The anchor Feature Set is the Feature Set that contains the superset of all primary key
        columns across all Feature Sets from all Features provided. If more than 1 Feature Set has the superset of
        all Feature Sets, the Feature Set with the most primary keys is selected. If more than 1 Feature Set has the same
        maximum number of primary keys, the Feature Set is chosen by alphabetical order (schema_name, table_name).

        :param features: List of Features or strings of feature names

            :NOTE:
                .. code-block:: text

                    The Features Sets which the list of Features come from must have common join keys,
                    otherwise the function will fail. If there is no common join key, it is recommended to
                    create a Training View to specify the join conditions.

        :param current_values_only: If you only want the most recent values of the features, set this to true. Otherwise, all history will be returned. Default False
        :param start_time: How far back in history you want Feature values. If not specified (and current_values_only is False), all history will be returned.
            This parameter only takes effect if current_values_only is False.
        :param end_time: The most recent values for each selected Feature. This will be the cutoff time, such that any Feature values that
            were updated after this point in time won't be selected. If not specified (and current_values_only is False),
            Feature values up to the moment in time you call the function (now) will be retrieved. This parameter
            only takes effect if current_values_only is False.
        :return: Spark DF
        """
        # Get List[Feature]
        features = self._process_features(features)

        # Get the Feature Sets
        fsets = self.get_feature_sets(list({f.feature_set_id for f in features}))

        if current_values_only:
            sql = _generate_training_set_sql(features, fsets)
        else:
            temp_vw = _create_temp_training_view(features, fsets)
            sql = _generate_training_set_history_sql(temp_vw, features, fsets, start_time=start_time, end_time=end_time)

        # Here we create a null training view and pass it into the training set. We do this because this special kind
        # of training set isn't standard. It's not based on a training view, on primary key columns, a label column,
        # or a timestamp column . This is simply a joined set of features from different feature sets.
        # But we still want to track this in mlflow as a user may build and deploy a model based on this. So we pass in
        # a null training view that can be tracked with a "name" (although the name is None). This is a likely case
        # for (non time based) clustering use cases.
        null_tvw = TrainingView(pk_columns=[], ts_column=None, label_column=None, view_sql=None, name=None,
                                description=None)
        ts = TrainingSet(training_view=null_tvw, features=features, start_time=start_time, end_time=end_time)

        # If the user isn't getting historical values, that means there isn't really a start_time, as the user simply
        # wants the most up to date values of each feature. So we set start_time to end_time (which is datetime.today)
        # For metadata purposes
        if current_values_only:
            ts.start_time = ts.end_time

        if hasattr(self, 'mlflow_ctx') and not return_sql:
            self.mlflow_ctx._active_training_set: TrainingSet = ts
            ts._register_metadata(self.mlflow_ctx)
        return sql if return_sql else self.splice_ctx.df(sql)

    def get_training_set_from_view(self, training_view: str, features: Union[List[Feature], List[str]] = None,
                                   start_time: Optional[datetime] = None, end_time: Optional[datetime] = None,
                                   return_sql: bool = False) -> SparkDF or str:
        """
        Returns the training set as a Spark Dataframe from a Training View. When a user calls this function (assuming they have registered
        the feature store with mlflow using :py:meth:`~mlflow.register_feature_store` )
        the training dataset's metadata will be tracked in mlflow automatically. The following will be tracked:
        including:
            * Training View
            * Selected features
            * Start time
            * End time
        This tracking will occur in the current run (if there is an active run)
        or in the next run that is started after calling this function (if no run is currently active).

        :param training_view: (str) The name of the registered training view
        :param features: (List[str] OR List[Feature]) the list of features from the feature store to be included in the training.
            If a list of strings is passed in it will be converted to a list of Feature. If not provided will return all available features.

            :NOTE:
                .. code-block:: text

                    This function will error if the view SQL is missing a join key required to retrieve the
                    desired features

        :param start_time: (Optional[datetime]) The start time of the query (how far back in the data to start). Default None

            :NOTE:
                .. code-block:: text

                    If start_time is None, query will start from beginning of history

        :param end_time: (Optional[datetime]) The end time of the query (how far recent in the data to get). Default None

            :NOTE:
                .. code-block:: text

                    If end_time is None, query will get most recently available data

        :param return_sql: (Optional[bool]) Return the SQL statement (str) instead of the Spark DF. Defaults False
        :return: Optional[SparkDF, str] The Spark dataframe of the training set or the SQL that is used to generate it (for debugging)
        """

        # Get features as list of Features
        features = self._process_features(features) if features else self.get_training_view_features(training_view)

        # Get List of necessary Feature Sets
        feature_set_ids = list({f.feature_set_id for f in features})  # Distinct set of IDs
        feature_sets = self.get_feature_sets(feature_set_ids)

        # Get training view information (view primary key column(s), inference ts column, )
        tvw = self.get_training_view(training_view)
        # Generate the SQL needed to create the dataset
        sql = _generate_training_set_history_sql(tvw, features, feature_sets, start_time=start_time, end_time=end_time)

        # Link this to mlflow for model deployment
        if hasattr(self, 'mlflow_ctx') and not return_sql:
            ts = TrainingSet(training_view=tvw, features=features,
                             start_time=start_time, end_time=end_time)
            self.mlflow_ctx._active_training_set: TrainingSet = ts
            ts._register_metadata(self.mlflow_ctx)

        return sql if return_sql else self.splice_ctx.df(sql)

    def list_training_sets(self) -> Dict[str, Optional[str]]:
        """
        Returns a dictionary a training sets available, with the map name -> description. If there is no description,
        the value will be an emtpy string

        :return: Dict[str, Optional[str]]
        """
        raise NotImplementedError("To see available training views, run fs.describe_training_views()")

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
        fset = FeatureSet(splice_ctx=self.splice_ctx, schema_name=schema_name, table_name=table_name,
                          primary_keys=primary_keys,
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

        if not re.match('^[A-Za-z][A-Za-z0-9_]*$', name, re.IGNORECASE):
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
        :param feature_type: splicemachine.features.FeatureType of the feature. The available types are from the FeatureType class: FeatureType.[categorical, ordinal, continuous].
            You can see available feature types by running

            .. code-block:: python

                    from splicemachine.features import FeatureType
                    print(FeatureType.get_valid())

        :param desc: The (optional) feature description (default None)
        :param tags: (optional) List of (str) tag words (default None)
        :return: Feature created
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
        return f
        # TODO: Backfill the feature

    def _validate_training_view(self, name, sql, join_keys, label_col=None):
        """
        Validates that the training view doesn't already exist.

        :param name: The training view name
        :param sql: The training view provided SQL
        :param join_keys: The provided join keys when creating the training view
        :param label_col: The label column
        :return:
        """
        # Validate name doesn't exist
        assert len(self.get_training_views(_filter={'name': name})) == 0, f"Training View {name} already exists!"

        # Column comparison
        # Lazily evaluate sql resultset, ensure that the result contains all columns matching pks, join_keys, tscol and label_col
        from py4j.protocol import Py4JJavaError
        try:
            valid_df = self.splice_ctx.df(sql)
        except Py4JJavaError as e:
            if 'SQLSyntaxErrorException' in str(e.java_exception):
                raise SpliceMachineException(f'The provided SQL is incorrect. The following error was raised during '
                                             f'validation:\n\n{str(e.java_exception)}') from None
            raise e

        # Ensure the label column specified is in the output of the SQL
        if label_col: assert label_col in valid_df.columns, f"Provided label column {label_col} is not available in the provided SQL"
        # Confirm that all join_keys provided correspond to primary keys of created feature sets
        pks = set(i[0].upper() for i in self.splice_ctx.df(SQL.get_fset_primary_keys).collect())
        missing_keys = set(i.upper() for i in join_keys) - pks
        assert not missing_keys, f"Not all provided join keys exist. Remove {missing_keys} or " \
                                 f"create a feature set that uses the missing keys"

    def create_training_view(self, name: str, sql: str, primary_keys: List[str], join_keys: List[str],
                             ts_col: str, label_col: Optional[str] = None, replace: Optional[bool] = False,
                             desc: Optional[str] = None, verbose=False) -> None:
        """
        Registers a training view for use in generating training SQL

        :param name: The training set name. This must be unique to other existing training sets unless replace is True
        :param sql: (str) a SELECT statement that includes:
            * the primary key column(s) - uniquely identifying a training row/case
            * the inference timestamp column - timestamp column with which to join features (temporal join timestamp)
            * join key(s) - the references to the other feature tables' primary keys (ie customer_id, location_id)
            * (optionally) the label expression - defining what the training set is trying to predict
        :param primary_keys: (List[str]) The list of columns from the training SQL that identify the training row
        :param ts_col: The timestamp column of the training SQL that identifies the inference timestamp
        :param label_col: (Optional[str]) The optional label column from the training SQL.
        :param replace: (Optional[bool]) Whether to replace an existing training view
        :param join_keys: (List[str]) A list of join keys in the sql that are used to get the desired features in
            get_training_set
        :param desc: (Optional[str]) An optional description of the training set
        :param verbose: Whether or not to print the SQL before execution (default False)
        :return:
        """
        assert name != "None", "Name of training view cannot be None!"
        self._validate_training_view(name, sql, join_keys, label_col)
        # register_training_view()
        label_col = f"'{label_col}'" if label_col else "NULL"  # Formatting incase NULL
        train_sql = SQL.training_view.format(name=name, desc=desc or 'None Provided', sql_text=sql, ts_col=ts_col,
                                             label_col=label_col)
        print('Building training sql...')
        if verbose: print('\t', train_sql)
        self.splice_ctx.execute(train_sql)
        print('Done.')

        # Get generated view ID
        vid = self.get_training_view_id(name)

        print('Creating Join Keys')
        for i in join_keys:
            key_sql = SQL.training_view_keys.format(view_id=vid, key_column=i.upper(), key_type='J')
            print(f'\tCreating Join Key {i}...')
            if verbose: print('\t', key_sql)
            self.splice_ctx.execute(key_sql)
        print('Done.')
        print('Creating Training View Primary Keys')
        for i in primary_keys:
            key_sql = SQL.training_view_keys.format(view_id=vid, key_column=i.upper(), key_type='P')
            print(f'\tCreating Primary Key {i}...')
            if verbose: print('\t', key_sql)
            self.splice_ctx.execute(key_sql)
        print('Done.')

    def _process_features(self, features: List[Union[Feature, str]]) -> List[Feature]:
        """
        Process a list of Features parameter. If the list is strings, it converts them to Features, else returns itself

        :param features: The list of Feature names or Feature objects
        :return: List[Feature]
        """
        feat_str = [f for f in features if isinstance(f, str)]
        str_to_feat = self.get_features_by_name(names=feat_str, as_list=True) if feat_str else []
        all_features = str_to_feat + [f for f in features if not isinstance(f, str)]
        assert all(
            [isinstance(i, Feature) for i in all_features]), "It seems you've passed in Features that are neither" \
                                                             " a feature name (string) or a Feature object"
        return all_features

    def deploy_feature_set(self, schema_name, table_name):
        """
        Deploys a feature set to the database. This persists the feature stores existence.
        As of now, once deployed you cannot delete the feature set or add/delete features.
        The feature set must have already been created with :py:meth:`~features.FeatureStore.create_feature_set`

        :param schema_name: The schema of the created feature set
        :param table_name: The table of the created feature set
        """
        try:
            fset = self.get_feature_sets(_filter={'schema_name': schema_name, 'table_name': table_name})[0]
        except:
            raise SpliceMachineException(
                f"Cannot find feature set {schema_name}.{table_name}. Ensure you've created this"
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
            print('-' * 23)
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
        if not fset: raise SpliceMachineException(
            f"Feature Set {schema_name}.{table_name} not found. Check name and try again.")
        fset = fset[0]
        print(f'{fset.schema_name}.{fset.table_name} - {fset.description}')
        print('Primary keys:', fset.primary_keys)
        print('\nAvailable features:')
        display(pd.DataFrame(f.__dict__ for f in fset.get_features()))

    def describe_training_views(self) -> None:
        """
        Prints out a description of all training views, the ID, name, description and optional label

        :param training_view: The training view name
        :return: None
        """
        print('Available training views')
        for tcx in self.get_training_views():
            print('-' * 23)
            self.describe_training_view(tcx.name)

    def describe_training_view(self, training_view: str) -> None:
        """
        Prints out a description of a given training view, the ID, name, description and optional label

        :param training_view: The training view name
        :return: None
        """
        tcx = self.get_training_views(_filter={'name': training_view})
        if not tcx: raise SpliceMachineException(f"Training view {training_view} not found. Check name and try again.")
        tcx = tcx[0]
        print(f'ID({tcx.view_id}) {tcx.name} - {tcx.description} - LABEL: {tcx.label_column}')
        print(f'Available features in {tcx.name}:')
        feats: List[Feature] = self.get_training_view_features(tcx.name)
        # Grab the feature set info and their corresponding names (schema.table) for the display table
        feat_sets: List[FeatureSet] = self.get_feature_sets(feature_set_ids=[f.feature_set_id for f in feats])
        feat_sets: Dict[int, str] = {fset.feature_set_id: f'{fset.schema_name}.{fset.table_name}' for fset in feat_sets}
        for f in feats:
            f.feature_set_name = feat_sets[f.feature_set_id]
        col_order = ['name', 'description', 'feature_data_type', 'feature_set_name', 'feature_type', 'tags',
                     'last_update_ts',
                     'last_update_username', 'compliance_level', 'feature_set_id', 'feature_id']
        display(pd.DataFrame(f.__dict__ for f in feats)[col_order])

    def set_feature_description(self):
        raise NotImplementedError

    def _retrieve_training_set_from_deployment(self, schema_name, table_name):
        metadata = self._retrieve_training_set_metadata_from_deployement(schema_name, table_name)
        features = metadata['FEATURES'].split(',')
        tv_name = metadata['NAME']
        start_time = metadata['TRAINING_SET_START_TS']
        end_time = metadata['TRAINING_SET_END_TS']
        if (tv_name):
            training_set_df = self.get_training_set_from_view(training_view=tv_name, start_time=start_time,
                                                              end_time=end_time, features=features)
        else:
            training_set_df = self.get_training_set(features=features, start_time=start_time, end_time=end_time)
        return training_set_df

    def _retrieve_model_data_sets(self, schema_name, table_name):
        training_set_df = self._retrieve_training_set_from_deployment(schema_name, table_name)
        model_table_df = self.splice_ctx.df(f'SELECT * FROM {schema_name}.{table_name}')
        return training_set_df, model_table_df

    def _retrieve_training_set_metadata_from_deployement(self, schema_name: str, table_name: str):
        sql = SQL.get_deployment_metadata.format(schema_name=schema_name, table_name=table_name)
        deploy_df = self.splice_ctx.df(sql).collect()
        cnt = len(deploy_df)
        if (cnt == 1):
            return deploy_df[0]

    def _calculate_bounds(self, df, column_name):
        """
        Calculates outlier bounds based on interquartile range of distribution of values in column 'column_name'
        from data set in data frame 'df'.
        :param df: data frame containing data to be analyzed
        :param column_name: column name to analyze
        :return: dictionary with keys min, max, q1 and q3 keys and corresponding values for outlier minimum, maximum
        and 25th and 75th percentile values (q1,q3)
        """
        bounds = dict(zip(["q1", "q3"], df.approxQuantile(column_name, [0.25, 0.75], 0)))
        iqr = bounds['q3'] - bounds['q1']
        bounds['min'] = bounds['q1'] - (iqr * 1.5)
        bounds['max'] = bounds['q3'] + (iqr * 1.5)
        return bounds

    def _remove_outliers(self, df, column_name):
        '''
        Calculates outlier bounds no distribution of 'column_name' values and returns a filtered data frame without
        outliers in the specified column.
        :param df: data frame with data to remove outliers from
        :param column_name: name of column to remove outliers from
        :return: input data frame filtered to remove outliers
        '''
        import pyspark.sql.functions as f
        bounds = self._calculate_bounds(df, column_name)
        return df.filter((f.col(column_name) >= bounds['min']) & (f.col(column_name) <= bounds['max']))

    def _add_feature_plot(self, ax, train_df, model_df, feature, n_bins):
        '''
        Adds a distplot of the outlier free feature values from both train_df and model_df data frames which both
        contain the feature.
        :param ax: target subplot for chart
        :param train_df: training data containing feature of interest
        :param model_df: model input data also containing feature of interest
        :param feature: name of feature to display in distribution histogram
        :param n_bins: number of bins to use in histogram plot
        :return: None
        '''
        from pyspark_dist_explore import distplot
        import pyspark.sql.functions as f
        distplot(ax, [self._remove_outliers(train_df.select(f.col(feature).alias('training')), 'training'),
                      self._remove_outliers(model_df.select(f.col(feature).alias('model')), 'model')], bins=n_bins)
        ax.set_title(feature)
        ax.legend()

    def display_model_feature_drift(self, schema_name, table_name):
        """
        Displays feature by feature comparison between the training set of the deployed model and the input feature
        values used with the model since deployment.
        :param schema_name: name of database schema where model table is deployed
        :param table_name: name of the model table
        :return: None
        """
        from matplotlib.pyplot import show, subplots
        metadata = self._retrieve_training_set_metadata_from_deployement(schema_name, table_name)
        if metadata:
            features = metadata['FEATURES'].split(',')
            training_set_df, model_table_df = self._retrieve_model_data_sets(schema_name, table_name)
            final_features = [f for f in features if f in model_table_df.columns]
            # prep plot area
            n_bins = 15
            num_features = len(final_features)
            n_rows = int(num_features / 5)
            if num_features % 5 > 0:
                n_rows = n_rows + 1
            fig, axes = subplots(nrows=n_rows, ncols=5, figsize=(30, 10 * n_rows))
            axes = axes.flatten()
            # calculate combined plots for each feature
            for plot, f in enumerate(final_features):
                self._add_feature_plot(axes[plot], training_set_df, model_table_df, f, n_bins)
            show()
        else:
            print(f"Could not find deployment for model table {schema_name}.{table_name}")

    def _datetime_range(self, start: datetime, end: datetime, number: int):
        """
        Subdivides the time frame defined by 'start' and 'end' parameters into 'number' equal time frames.
        :param start: start date time
        :param end: end date time
        :param number: number of time frames to split into
        :return: list of start/end date times
        """
        from datetime import datetime
        from itertools import count, islice
        start_secs = (start - datetime(1970, 1, 1)).total_seconds()
        end_secs = (end - datetime(1970, 1, 1)).total_seconds()
        dates = [datetime.fromtimestamp(el) for el in
                 islice(count(start_secs, (end_secs - start_secs) / number), number + 1)]
        return zip(dates, dates[1:])

    def display_model_drift(self, schema_name: str, table_name: str, time_intervals: int,
                            start_time: datetime = None, end_time: datetime = None):
        """
        Displays as many as 'time_intervals' plots showing the distribution of the model prediction within each time
        period. Time periods are equal periods of time where predictions are present in the model table
        'schema_name'.'table_name'. Model predictions are first filtered to only those occurring after 'start_time' if
        specified and before 'end_time' if specified.
        :param schema_name: schema where the model table resides
        :param table_name: name of the model table
        :param time_intervals: number of time intervals to plot
        :param start_time: if specified, filters to only show predictions occurring after this date/time
        :param end_time: if specified, filters to only show predictions occurring before this date/time
        :return: None
        """
        from datetime import datetime
        import matplotlib.pyplot as plt
        from pyspark_dist_explore import distplot
        import pyspark.sql.functions as f

        # set default timeframe if not specified
        if not start_time:
            start_time = datetime(1900, 1, 1, 0, 0, 0)
        if not end_time:
            end_time = datetime.now()
        # retrieve predictions the model has made over time
        sql = SQL.get_model_predictions.format(schema_name=schema_name, table_name=table_name, start_time=start_time,
                                               end_time=end_time)
        model_table_df = self.splice_ctx.df(sql)
        min_ts = model_table_df.first()['EVAL_TIME']
        max_ts = model_table_df.orderBy(f.col("EVAL_TIME").desc()).first()['EVAL_TIME']

        if max_ts > min_ts:
            intervals = self._datetime_range(min_ts, max_ts, time_intervals)
            n_rows = int(time_intervals / 5)
            if time_intervals % 5 > 0:
                n_rows = n_rows + 1
            fig, axes = plt.subplots(nrows=n_rows, ncols=5, figsize=(30, 10 * n_rows))
            axes = axes.flatten()
            for i, time_int in enumerate(intervals):
                df = model_table_df.filter((f.col('EVAL_TIME') >= time_int[0]) & (f.col('EVAL_TIME') < time_int[1]))
                distplot(axes[i], [self._remove_outliers(df.select(f.col('PREDICTION')), 'PREDICTION')], bins=15)
                axes[i].set_title(f"{time_int[0]}")
                axes[i].legend()
        else:
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
            distplot(axes, [self._remove_outliers(model_table_df.select(f.col('PREDICTION')), 'PREDICTION')], bins=15)
            axes.set_title(f"Predictions at {min_ts}")
            axes.legend()

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

        v = VectorAssembler(inputCols=all_features, outputCol='features', handleInvalid='keep')
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

    def __prune_features_for_elimination(self, features) -> List[Feature]:
        """
        Removes incompatible features from the provided list if they are not compatible with SparkML modeling

        :param features: List[Feature] the provided list
        :return: List[Features] the pruned list
        """
        from splicemachine.spark.constants import SQL_MODELING_TYPES
        invalid_features = {f for f in features if f.feature_data_type not in SQL_MODELING_TYPES}
        valid_features = list(set(features) - invalid_features)
        if invalid_features: print('The following features are invalid for modeling based on their Data Types:\n')
        for f in invalid_features:
            print(f.name, f.feature_data_type)
        return valid_features

    def run_feature_elimination(self, df, features: List[Union[str, Feature]], label: str = 'label', n: int = 10,
                                verbose: int = 0, model_type: str = 'classification', step: int = 1,
                                log_mlflow: bool = False, mlflow_run_name: str = None,
                                return_importances: bool = False):

        """
        Runs feature elimination using a Spark decision tree on the dataframe passed in. Optionally logs results to mlflow

        :param df: The dataframe with features and label
        :param features: The list of feature names (or Feature objects) to run elimination on
        :param label: the label column names
        :param n: The number of features desired. Default 10
        :param verbose: The level of verbosity. 0 indicated no printing. 1 indicates printing remaining features after
            each round. 2 indicates print features and relative importances after each round. Default 0
        :param model_type: Whether the model to test with will be a regression or classification model. Default classification
        :param log_mlflow: Whether or not to log results to mlflow as nested runs. Default false
        :param mlflow_run_name: The name of the parent run under which all subsequent runs will live. The children run
            names will be {mlflow_run_name}_{num_features}_features. ie testrun_5_features, testrun_4_features etc
        :return:
        """

        train_df = df
        features = self._process_features(features)
        remaining_features = self.__prune_features_for_elimination(features)
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
                                                                model.transform(train_df), "features").head(
                num_features)
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
