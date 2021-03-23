from typing import List, Dict, Optional, Union
from datetime import datetime
import re

from IPython.display import display
import pandas as pd
from pandas import DataFrame as PandasDF

from pyspark.sql.dataframe import DataFrame as SparkDF
import pyspark.sql.functions as psf
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import StringIndexer, VectorAssembler

from splicemachine import SpliceMachineException
from splicemachine.spark import PySpliceContext
from splicemachine.features import Feature, FeatureSet
from .training_set import TrainingSet
from .utils.drift_utils import (add_feature_plot, remove_outliers, datetime_range_split, build_feature_drift_plot, build_model_drift_plot)
from .utils.training_utils import (dict_to_lower, _generate_training_set_history_sql,
                                   _generate_training_set_sql, _create_temp_training_view)
from .constants import SQL, FeatureType
from .training_view import TrainingView
from .utils.http_utils import RequestType, make_request, _get_feature_store_url, Endpoints, _get_credentials
import requests
import warnings
from requests.auth import HTTPBasicAuth

class FeatureStore:
    def __init__(self, splice_ctx: PySpliceContext = None) -> None:
        self.splice_ctx = splice_ctx
        self.mlflow_ctx = None
        self.feature_sets = []  # Cache of newly created feature sets
        self._FS_URL = _get_feature_store_url()
        if not self._FS_URL: warnings.warn(
            "Uh Oh! FS_URL variable was not found... you should call 'fs.set_feature_store_url(<url>)' before doing anything.")
        self._basic_auth = None
        self.__try_auto_login()

    def register_splice_context(self, splice_ctx: PySpliceContext) -> None:
        self.splice_ctx = splice_ctx

    def get_feature_sets(self, feature_set_names: List[str] = None) -> List[FeatureSet]:
        """
        Returns a list of available feature sets
        
        :param feature_set_names: A list of feature set names in the format '{schema_name}.{table_name}'. If none will return all FeatureSets
        :return: List[FeatureSet] the list of Feature Sets
        """

        r = make_request(self._FS_URL, Endpoints.FEATURE_SETS, RequestType.GET, self._basic_auth,
                         { "name": feature_set_names } if feature_set_names else None)
        return [FeatureSet(**fs) for fs in r]

    def remove_training_view(self, name: str):
        """
        This removes a training view if it is not being used by any currently deployed models.
        NOTE: Once this training view is removed, you will not be able to deploy any models that were trained using this
        view

        :param name: The view name
        """
        print(f"Removing Training View {name}")
        make_request(self._FS_URL, Endpoints.TRAINING_VIEWS, RequestType.DELETE, self._basic_auth, { "name": name })
        print('Done')

    def get_summary(self) -> TrainingView:
        """
        This function returns a summary of the feature store including:
            * Number of feature sets
            * Number of deployed feature sets
            * Number of features
            * Number of deployed features
            * Number of training sets
            * Number of training views
            * Number of associated models - this is a count of the MLManager.RUNS table where the `splice.model_name` tag is set and the `splice.feature_store.training_set` parameter is set
            * Number of active (deployed) models (that have used the feature store for training)
            * Number of pending feature sets - this will will require a new table `featurestore.pending_feature_set_deployments` and it will be a count of that
        """

        r = make_request(self._FS_URL, Endpoints.SUMMARY, RequestType.GET, self._basic_auth)
        return r

    def get_training_view(self, training_view: str) -> TrainingView:
        """
        Gets a training view by name

        :param training_view: Training view name
        :return: TrainingView
        """

        r = make_request(self._FS_URL, Endpoints.TRAINING_VIEWS, RequestType.GET, self._basic_auth, { "name": training_view })
        return TrainingView(**r[0])

    def get_training_views(self, _filter: Dict[str, Union[int, str]] = None) -> List[TrainingView]:
        """
        Returns a list of all available training views with an optional filter

        :param _filter: Dictionary container the filter keyword (label, description etc) and the value to filter on
            If None, will return all TrainingViews
        :return: List[TrainingView]
        """

        r = make_request(self._FS_URL, Endpoints.TRAINING_VIEWS, RequestType.GET, self._basic_auth)
        return [TrainingView(**tv) for tv in r]

    def get_training_view_id(self, name: str) -> int:
        """
        Returns the unique view ID from a name

        :param name: The training view name
        :return: The training view id
        """
        # return self.splice_ctx.df(SQL.get_training_view_id.format(name=name)).collect()[0][0]
        r = make_request(self._FS_URL, Endpoints.TRAINING_VIEW_ID, RequestType.GET, self._basic_auth, { "name": name })
        return int(r)

    def get_features_by_name(self, names: Optional[List[str]] = None, as_list=False) -> Union[List[Feature], SparkDF]:
        """
        Returns a dataframe or list of features whose names are provided

        :param names: The list of feature names
        :param as_list: Whether or not to return a list of features. Default False
        :return: SparkDF or List[Feature] The list of Feature objects or Spark Dataframe of features and their metadata. Note, this is not the Feature
        values, simply the describing metadata about the features. To create a training dataset with Feature values, see
        :py:meth:`features.FeatureStore.get_training_set` or :py:meth:`features.FeatureStore.get_feature_dataset`
        """
        r = make_request(self._FS_URL, Endpoints.FEATURES, RequestType.GET, self._basic_auth, { "name": names })
        return [Feature(**f) for f in r] if as_list else pd.DataFrame.from_dict(r)

    def get_feature_vector(self, features: List[Union[str, Feature]],
                           join_key_values: Dict[str, str], return_sql=False) -> Union[str, PandasDF]:
        """
        Gets a feature vector given a list of Features and primary key values for their corresponding Feature Sets

        :param features: List of str Feature names or Features
        :param join_key_values: (dict) join key values to get the proper Feature values formatted as {join_key_column_name: join_key_value}
        :param return_sql: Whether to return the SQL needed to get the vector or the values themselves. Default False
        :return: Pandas Dataframe or str (SQL statement)
        """
        features = [f if isinstance(f, str) else f.__dict__ for f in features]
        r = make_request(self._FS_URL, Endpoints.FEATURE_VECTOR, RequestType.POST, self._basic_auth, 
            { "sql": return_sql }, { "features": features, "join_key_values": join_key_values })
        return r if return_sql else pd.DataFrame(r, index=[0])


    def get_feature_vector_sql_from_training_view(self, training_view: str, features: List[Union[str,Feature]]) -> str:
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
        features = [f if isinstance(f, str) else f.__dict__ for f in features]
        r = make_request(self._FS_URL, Endpoints.FEATURE_VECTOR_SQL, RequestType.POST, self._basic_auth, 
            { "view": training_view }, features)
        return r

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

        r = make_request(self._FS_URL, Endpoints.TRAINING_VIEW_FEATURES, RequestType.GET,
                         self._basic_auth, { "view": training_view })
        return [Feature(**f) for f in r]

    def get_feature_description(self):
        # TODO
        raise NotImplementedError

    def get_training_set(self, features: Union[List[Feature], List[str]], current_values_only: bool = False,
                         start_time: datetime = None, end_time: datetime = None, label: str = None, return_pk_cols: bool = False, 
                         return_ts_col: bool = False, return_sql: bool = False) -> SparkDF or str:
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
        :param label: An optional label to specify for the training set. If specified, the feature set of that feature
            will be used as the "anchor" feature set, meaning all point in time joins will be made to the timestamps of
            that feature set. This feature will also be recorded as a "label" feature for this particular training set
            (but not others in the future, unless this label is again specified).
        :param return_pk_cols: bool Whether or not the returned sql should include the primary key column(s)
        :param return_ts_cols: bool Whether or not the returned sql should include the timestamp column
        :return: Spark DF
        """
        features = [f if isinstance(f, str) else f.__dict__ for f in features]
        r = make_request(self._FS_URL, Endpoints.TRAINING_SETS, RequestType.POST, self._basic_auth, 
                        { "current": current_values_only, "label": label, "pks": return_pk_cols, "ts": return_ts_col }, 
                        { "features": features, "start_time": start_time, "end_time": end_time })
        create_time = r['metadata']['training_set_create_ts']
        sql = r['sql']
        tvw = TrainingView(**r['training_view'])
        features = [Feature(**f) for f in r['features']]
        # Here we create a null training view and pass it into the training set. We do this because this special kind
        # of training set isn't standard. It's not based on a training view, on primary key columns, a label column,
        # or a timestamp column . This is simply a joined set of features from different feature sets.
        # But we still want to track this in mlflow as a user may build and deploy a model based on this. So we pass in
        # a null training view that can be tracked with a "name" (although the name is None). This is a likely case
        # for (non time based) clustering use cases.
        # null_tvw = TrainingView(pk_columns=[], ts_column=None, label_column=None, view_sql=None, name=None,
        #                         description=None)
        # ts = TrainingSet(training_view=null_tvw, features=features, start_time=start_time, end_time=end_time)

        # If the user isn't getting historical values, that means there isn't really a start_time, as the user simply
        # wants the most up to date values of each feature. So we set start_time to end_time (which is datetime.today)

        if self.mlflow_ctx and not return_sql:
            self.link_training_set_to_mlflow(features, create_time, start_time, end_time, tvw)
        return sql if return_sql else self.splice_ctx.df(sql)

    def get_training_set_from_view(self, training_view: str, features: Union[List[Feature], List[str]] = None,
                                   start_time: Optional[datetime] = None, end_time: Optional[datetime] = None,
                                   return_pk_cols: bool = False, return_ts_col: bool = False, return_sql: bool = False) -> SparkDF or str:
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

        :param return_pk_cols: bool Whether or not the returned sql should include the primary key column(s)
        :param return_ts_cols: bool Whether or not the returned sql should include the timestamp column
        :param return_sql: (Optional[bool]) Return the SQL statement (str) instead of the Spark DF. Defaults False
        :return: Optional[SparkDF, str] The Spark dataframe of the training set or the SQL that is used to generate it (for debugging)
        """

        # # Generate the SQL needed to create the dataset
        features = [f if isinstance(f, str) else f.__dict__ for f in features] if features else None
        r = make_request(self._FS_URL, Endpoints.TRAINING_SET_FROM_VIEW, RequestType.POST, self._basic_auth, 
                        { "view": training_view, "pks": return_pk_cols, "ts": return_ts_col }, 
                        { "features": features, "start_time": start_time, "end_time": end_time })
        sql = r["sql"]
        tvw = TrainingView(**r["training_view"])
        features = [Feature(**f) for f in r["features"]]
        create_time = r['metadata']['training_set_create_ts']

        # Link this to mlflow for model deployment
        if self.mlflow_ctx and not return_sql:
            self.link_training_set_to_mlflow(features, create_time, start_time, end_time, tvw)

        return sql if return_sql else self.splice_ctx.df(sql)

    def list_training_sets(self) -> Dict[str, Optional[str]]:
        """
        Returns a dictionary a training sets available, with the map name -> description. If there is no description,
        the value will be an emtpy string

        :return: Dict[str, Optional[str]]
        """
        raise NotImplementedError("To see available training views, run fs.describe_training_views()")


    def create_feature_set(self, schema_name: str, table_name: str, primary_keys: Dict[str, str],
                           desc: Optional[str] = None, features: Optional[List[Feature]] = None) -> FeatureSet:
        """
        Creates and returns a new feature set

        :param schema_name: The schema under which to create the feature set table
        :param table_name: The table name for this feature set
        :param primary_keys: The primary key column(s) of this feature set
        :param desc: The (optional) description
        :param features: An optional list of features. If provided, the Features will be created with the Feature Set
        :Example:
            .. code-block:: python

                from splicemachine.features import FeatureType, Feature
                f1 = Feature(
                    name='my_first_feature',
                    description='the first feature',
                    feature_data_type='INT',
                    feature_type=FeatureType.ordinal,
                    tags=['good_feature','a new tag', 'ordinal'],
                    attributes={'quality':'awesome'}
                )
                f2 = Feature(
                    name='my_second_feature',
                    description='the second feature',
                    feature_data_type='FLOAT',
                    feature_type=FeatureType.continuous,
                    tags=['not_as_good_feature','a new tag'],
                    attributes={'quality':'not as awesome'}
                )
                feats = [f1, f2]
                feature_set = fs.create_feature_set(
                    schema_name='splice',
                    table_name='foo',
                    primary_keys={'MOMENT_KEY':"INT"},
                    desc='test fset',
                    features=feats
                )

        :return: FeatureSet
        """
        # database stores object names in upper case
        schema_name = schema_name.upper()
        table_name = table_name.upper()

        features = [f.__dict__ for f in features] if features else None
        fset_dict = { "schema_name": schema_name,
                      "table_name": table_name,
                      "primary_keys": primary_keys,
                      "description": desc,
                      "features": features}

        print(f'Registering feature set {schema_name}.{table_name} in Feature Store')
        if features:
            print(f'Registering {len(features)} features for {schema_name}.{table_name} in the Feature Store')
        r = make_request(self._FS_URL, Endpoints.FEATURE_SETS, RequestType.POST, self._basic_auth, body=fset_dict)
        return FeatureSet(**r)

    def create_feature(self, schema_name: str, table_name: str, name: str, feature_data_type: str,
                       feature_type: str, desc: str = None, tags: List[str] = None, attributes: Dict[str, str] = None):
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
        :param attributes: (optional) Dict of (str) attribute key/value pairs (default None)
        :return: Feature created
        """
        # database stores object names in upper case
        schema_name = schema_name.upper()
        table_name = table_name.upper()

        assert feature_type in FeatureType.get_valid(), f"The feature_type {feature_type} in not valid. Valid feature " \
                                                        f"types include {FeatureType.get_valid()}. Use the FeatureType" \
                                                        f" class provided by splicemachine.features"

        f_dict = { "name": name, "description": desc or '', "feature_data_type": feature_data_type,
                    "feature_type": feature_type, "tags": tags, "attributes": attributes }
        print(f'Registering feature {name} in Feature Store')
        r = make_request(self._FS_URL, Endpoints.FEATURES, RequestType.POST, self._basic_auth, 
            { "schema": schema_name, "table": table_name }, f_dict)
        f = Feature(**r)
        return f
        # TODO: Backfill the feature

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

        tv_dict = { "name": name, "description": desc, "pk_columns": primary_keys, "ts_column": ts_col, "label_column": label_col,
                    "join_columns": join_keys, "sql_text": sql}
        make_request(self._FS_URL, Endpoints.TRAINING_VIEWS, RequestType.POST, self._basic_auth, body=tv_dict)

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

    def deploy_feature_set(self, schema_name: str, table_name: str):
        """
        Deploys a feature set to the database. This persists the feature stores existence.
        As of now, once deployed you cannot delete the feature set or add/delete features.
        The feature set must have already been created with :py:meth:`~features.FeatureStore.create_feature_set`

        :param schema_name: The schema of the created feature set
        :param table_name: The table of the created feature set
        """

        # database stores object names in upper case
        schema_name = schema_name.upper()
        table_name = table_name.upper()
        print(f'Deploying Feature Set {schema_name}.{table_name}')
        make_request(self._FS_URL, Endpoints.DEPLOY_FEATURE_SET, RequestType.POST, self._basic_auth, { "schema": schema_name, "table": table_name })
        print('Done')

    def describe_feature_sets(self) -> None:
        """
        Prints out a description of a all feature sets, with all features in the feature sets and whether the feature
        set is deployed

        :return: None
        """
        r = make_request(self._FS_URL, Endpoints.FEATURE_SET_DESCRIPTIONS, RequestType.GET, self._basic_auth)
        
        print('Available feature sets')
        for desc in r:
            features = [Feature(**feature) for feature in desc.pop('features')]
            fset = FeatureSet(**desc)
            print('-' * 23)
            self._feature_set_describe(fset, features)

    def describe_feature_set(self, schema_name: str, table_name: str) -> None:
        """
        Prints out a description of a given feature set, with all features in the feature set and whether the feature
        set is deployed

        :param schema_name: feature set schema name
        :param table_name: feature set table name
        :return: None
        """
        # database stores object names in upper case
        schema_name = schema_name.upper()
        table_name = table_name.upper()

        r = make_request(self._FS_URL, Endpoints.FEATURE_SET_DESCRIPTIONS, RequestType.GET, self._basic_auth,
                         params={'schema':schema_name, 'table':table_name})
        descs = r
        if not descs: raise SpliceMachineException(
            f"Feature Set {schema_name}.{table_name} not found. Check name and try again.")
        desc = descs[0]
        features = [Feature(**feature) for feature in desc.pop("features")]
        fset = FeatureSet(**desc)
        self._feature_set_describe(fset, features)

    def _feature_set_describe(self, fset: FeatureSet, features: List[Feature]):
        print(f'{fset.schema_name}.{fset.table_name} - {fset.description}')
        print('Primary keys:', fset.primary_keys)
        print('\nAvailable features:')
        display(pd.DataFrame(f.__dict__ for f in features))

    def describe_training_views(self) -> None:
        """
        Prints out a description of all training views, the ID, name, description and optional label

        :param training_view: The training view name
        :return: None
        """
        r = make_request(self._FS_URL, Endpoints.TRAINING_VIEW_DESCRIPTIONS, RequestType.GET, self._basic_auth)

        print('Available training views')
        for desc in r:
            features = [Feature(**f) for f in desc.pop('features')]
            tcx = TrainingView(**desc)
            print('-' * 23)
            self._training_view_describe(tcx, features)

    def describe_training_view(self, training_view: str) -> None:
        """
        Prints out a description of a given training view, the ID, name, description and optional label

        :param training_view: The training view name
        :return: None
        """

        r = make_request(self._FS_URL, Endpoints.TRAINING_VIEW_DESCRIPTIONS, RequestType.GET, self._basic_auth, {'name': training_view})
        descs = r
        if not descs: raise SpliceMachineException(f"Training view {training_view} not found. Check name and try again.")
        desc = descs[0]
        feats = [Feature(**f) for f in desc.pop('features')]
        tcx = TrainingView(**desc)
        self._training_view_describe(tcx, feats)

    def _training_view_describe(self, tcx: TrainingView, feats: List[Feature]):
        print(f'ID({tcx.view_id}) {tcx.name} - {tcx.description} - LABEL: {tcx.label_column}')
        print(f'Available features in {tcx.name}:')

        col_order = ['name', 'description', 'feature_data_type', 'feature_set_name', 'feature_type', 'tags',
                     'last_update_ts',
                     'last_update_username', 'compliance_level', 'feature_set_id', 'feature_id']
        display(pd.DataFrame(f.__dict__ for f in feats)[col_order])

    def set_feature_description(self):
        raise NotImplementedError

    def get_training_set_from_deployment(self, schema_name: str, table_name: str, label: str = None, 
                                        return_pk_cols: bool = False, return_ts_col: bool = False):
        """
        Reads Feature Store metadata to rebuild orginal training data set used for the given deployed model.
        :param schema_name: model schema name
        :param table_name: model table name
        :param label: An optional label to specify for the training set. If specified, the feature set of that feature
            will be used as the "anchor" feature set, meaning all point in time joins will be made to the timestamps of
            that feature set. This feature will also be recorded as a "label" feature for this particular training set
            (but not others in the future, unless this label is again specified).
        :param return_pk_cols: bool Whether or not the returned sql should include the primary key column(s)
        :param return_ts_cols: bool Whether or not the returned sql should include the timestamp column
        :return:
        """
        # database stores object names in upper case
        schema_name = schema_name.upper()
        table_name = table_name.upper()

        r = make_request(self._FS_URL, Endpoints.TRAINING_SET_FROM_DEPLOYMENT, RequestType.GET, self._basic_auth, 
            { "schema": schema_name, "table": table_name, "label": label, "pks": return_pk_cols, "ts": return_ts_col})
        
        metadata = r['metadata']
        sql = r['sql']

        tv_name = metadata['name']
        start_time = metadata['training_set_start_ts']
        end_time = metadata['training_set_end_ts']
        create_time = metadata['training_set_create_ts']

        tv = TrainingView(**r['training_view']) if 'training_view' in r else None
        features = [Feature(**f) for f in r['features']]

        if self.mlflow_ctx:
            self.link_training_set_to_mlflow(features, create_time, start_time, end_time, tv)
        return self.splice_ctx.df(sql)

    def remove_feature(self, name: str):
        """
        Removes a feature. This will run 2 checks.
            1. See if the feature exists.
            2. See if the feature belongs to a feature set that has already been deployed.
            If either of these are true, this function will throw an error explaining which check has failed
            :param name: feature name
            :return:
        """
        print(f"Removing feature {name}")
        make_request(self._FS_URL, Endpoints.FEATURES, RequestType.DELETE, self._basic_auth, { "name": name })
        print('Done')

    def get_deployments(self, schema_name: str = None, table_name: str = None, training_set: str = None):
        """
        Returns a list of all (or specified) available deployments
        :param schema_name: model schema name
        :param table_name: model table name
        :param training_set: training set name
        :return: List[Deployment] the list of Deployments as dicts
        """
        return make_request(self._FS_URL, Endpoints.DEPLOYMENTS, RequestType.GET, self._basic_auth, 
            { 'schema': schema_name, 'table': table_name, 'name': training_set })
      
    def get_training_set_features(self, training_set: str = None):
        """
        Returns a list of all features from an available Training Set, as well as details about that Training Set
        :param training_set: training set name
        :return: TrainingSet as dict
        """
        r = make_request(self._FS_URL, Endpoints.TRAINING_SET_FEATURES, RequestType.GET, self._basic_auth, 
            { 'name': training_set })
        r['features'] = [Feature(**f) for f in r['features']]
        return r

    def remove_feature_set(self, schema: str, table: str, purge: bool = False) -> None:
        """
        Deletes a feature set if appropriate. You can currently delete a feature set in two scenarios:
        1. The feature set has not been deployed
        2. The feature set has been deployed, but not linked to any training sets

        If both of these conditions are false, this will fail.

        Optionally set purge=True to force delete the feature set and all of the associated Training Sets using the
        Feature Set. ONLY USE IF YOU KNOW WHAT YOU ARE DOING. This will delete Training Sets, but will still fail if
        there is an active deployment with this feature set. That cannot be overwritten

        :param schema: The Feature Set Schema
        :param table: The Feature Set Table
        :param purge: Whether to force delete training sets that use the feature set (that are not used in deployments)
        """
        if purge:
            warnings.warn("You've set purge=True, I hope you know what you are doing! This will delete any dependent"
                          " Training Sets (except ones used in an active model deployment)")
        print(f'Removing Feature Set {schema}.{table}')
        make_request(self._FS_URL, Endpoints.FEATURE_SETS,
                     RequestType.DELETE, self._basic_auth, { "schema": schema, "table":table, "purge": purge })
        print('Done')

    def _retrieve_model_data_sets(self, schema_name: str, table_name: str):
        """
        Returns the training set dataframe and model table dataframe for a given deployed model.
        :param schema_name: model schema name
        :param table_name: model table name
        :return:
        """
        # database stores object names in upper case
        schema_name = schema_name.upper()
        table_name = table_name.upper()

        training_set_df = self.get_training_set_from_deployment(schema_name, table_name)
        model_table_df = self.splice_ctx.df(f'SELECT * FROM {schema_name}.{table_name}')
        return training_set_df, model_table_df

    def _retrieve_training_set_metadata_from_deployement(self, schema_name: str, table_name: str):
        """
        Reads Feature Store metadata to retrieve definition of training set used to train the specified model.
        :param schema_name: model schema name
        :param table_name: model table name
        :return:
        """
        # database stores object names in upper case
        schema_name = schema_name.upper()
        table_name = table_name.upper()

        sql = SQL.get_deployment_metadata.format(schema_name=schema_name, table_name=table_name)
        deploy_df = self.splice_ctx.df(sql).collect()
        cnt = len(deploy_df)
        if cnt == 1:
            return deploy_df[0]

    def display_model_feature_drift(self, schema_name: str, table_name: str):
        """
        Displays feature by feature comparison between the training set of the deployed model and the input feature
        values used with the model since deployment.
        :param schema_name: name of database schema where model table is deployed
        :param table_name: name of the model table
        :return: None
        """
        # database stores object names in upper case
        schema_name = schema_name.upper()
        table_name = table_name.upper()

        metadata = self._retrieve_training_set_metadata_from_deployement(schema_name, table_name)
        if not metadata:
            raise SpliceMachineException(f"Could not find deployment for model table {schema_name}.{table_name}") from None
        training_set_df, model_table_df = self._retrieve_model_data_sets(schema_name, table_name)
        features = metadata['FEATURES'].split(',')
        build_feature_drift_plot(features, training_set_df, model_table_df)


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
        # database stores object names in upper case
        schema_name = schema_name.upper()
        table_name = table_name.upper()

        # set default timeframe if not specified
        if not start_time:
            start_time = datetime(1900, 1, 1, 0, 0, 0)
        if not end_time:
            end_time = datetime.now()
        # retrieve predictions the model has made over time
        sql = SQL.get_model_predictions.format(schema_name=schema_name, table_name=table_name,
                                               start_time=start_time, end_time=end_time)
        model_table_df = self.splice_ctx.df(sql)
        build_model_drift_plot(model_table_df, time_intervals)


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
        try:
            if self.mlflow_ctx.active_run():
                self.mlflow_ctx.start_run(run_name=name)
            for r in range(rounds):
                with self.mlflow_ctx.start_run(run_name=f'Round {r}', nested=True):
                    self.mlflow_ctx.log_metrics(mlflow_results[r])
        finally:
            self.mlflow_ctx.end_run()


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

        if log_mlflow and self.mlflow_ctx:
            run_name = mlflow_run_name or f'feature_elimination_{label}'
            self.__log_mlflow_results(run_name, rnd, mlflow_results)

        return remaining_features, feature_importances.reset_index(
            drop=True) if return_importances else remaining_features

    def link_training_set_to_mlflow(self, features: Union[List[Feature], List[str]], create_time: datetime, start_time: datetime = None, 
                                    end_time: datetime = None, tvw: TrainingView = None, current_values_only: bool = False):
        if not tvw:
            tvw = TrainingView(pk_columns=[], ts_column=None, label_column=None, view_sql=None, name=None,
                                description=None)
        ts = TrainingSet(training_view=tvw, features=features, create_time=create_time,
                        start_time=start_time, end_time=end_time)

        # For metadata purposes
        if current_values_only:
            ts.start_time = ts.end_time

        self.mlflow_ctx._active_training_set: TrainingSet = ts
        ts._register_metadata(self.mlflow_ctx)

    
    def set_feature_store_url(self, url: str):
        self._FS_URL = url

    def login_fs(self, username, password):
        self._basic_auth = HTTPBasicAuth(username, password)

    def __try_auto_login(self):
        """
        Tries to login the user  automatically. This will only work if the user is not
        using the cloud service.

        :return: None
        """
        user, password = _get_credentials()
        if user and password:
            self.login_fs(user, password)
