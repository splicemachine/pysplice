# Installing dependancies
import os
commands_pip = [ # so you can easily rerun :)
"""curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py""",
"""python get-pip.py --user""",
"""export PATH=$PATH:/zeppelin/.local/bin/"""
]

for cmd_pip in commands_pip:
    os.system(cmd_pip)

# Installing Dependencies
packages = """
pyspark_dist_explore
tqdm
statsmodels
"""

print('stats_requirements.txt')
print('-' * 50)
print(packages)
with open('requirements.txt', 'w') as pcks_file:
    pcks_file.write(packages)

os.system('~/.local/bin/pip install -r stats_requirements.txt --user')

import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels as sm
from numpy.linalg import eigh

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, ArrayType, IntegerType, StringType
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, Bucketizer
from pyspark.ml.feature import PCA
from pyspark import keyword_only

import pyspark_dist_explore as dist_explore #FIXME-- MUST INSTALL IN REQUIRMENTS
import matplotlib.pyplot as plt
from tqdm import tqdm


# Custom Transformers
class OneHotDummies(Transformer, HasInputCol, HasOutputCol):
    """Transformer to generate dummy columns for categorical variables as a part of a preprocessing pipeline

    Follows: https://spark.apache.org/docs/latest/ml-pipeline.html#transformers

    Arguments:
        Transformer {[object]} -- Inherited Classes
        HasInputCol {[object]} -- Inherited Classes
        HasOutputCol {[object]} -- Inherited Classes

    Returns:
        [pyspark DataFrame]
    """
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        """Assigns variables to parameters passed

        Keyword Arguments:
            inputCol {[string]} -- Sparse vector returned by OneHotEncoders(default: {None})
            outputCol {[string]} -- Column Name Base to be used in naming dummy columns (default: {None})
        """
        super(OneHotDummies, self).__init__()
        # kwargs = self._input_kwargs
        # self.setParams(**kwargs)
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.outcols = []
    @keyword_only
    def setParams(self, inputCol=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        """transform method-- iterates through the number of categorical values of a categorical variable and adds dummy columns for each of those categories

        For a string categorical column, include this transformer in the following workflow: StringIndexer -> OneHotEncoder -> OneHotDummies -> PCA/ [{Learning Algorithm}]

        Arguments:
            dataset {[pyspark DataFrame]} -- inputCol is the column  returned by by OneHotEncoders

        Returns:
            [pyspark DataFrame] -- original DataFrame with M additional columns where M = # of categories for this variable
        """
        out_col_suffix = self.outputCol # this is what I want to append to the column name
        col_name = self.inputCol

        out_col_base = col_name+out_col_suffix # this is the base for the n outputted columns

        # helper functions
        get_num_categories = F.udf(lambda x: int(x.size), IntegerType())
        get_active_index = F.udf(lambda x: int(x.indices[0]), IntegerType())
        check_active_index = F.udf(lambda active, i: int(active == i), IntegerType())


        num_categories = dataset.select(get_num_categories(col_name).alias('num_categories')).distinct() # this returns a dataframe
        if num_categories.count() == 1: # making sure all the sparse vectors have the same number of categories
            num_categories_int = num_categories.collect()[0]['num_categories'] # now this is an int

        dataset = dataset.withColumn('active_index', get_active_index(col_name))
        column_names = []
        for i in range(num_categories_int): # Now I'm going to make a column for each category
            column_name = out_col_base+'_'+str(i)
            dataset = dataset.withColumn(column_name, check_active_index(F.col('active_index'), F.lit(i)))
            column_names.append(column_name)

        dataset = dataset.drop('active_index')
        self.outcols = column_names
        return dataset

    def getOutCols(self):
        return self.outcols

class IndReconstructer(Transformer, HasInputCol, HasOutputCol):
    """Transformer to reconstruct String Index from OneHotDummy Columns. This can be used as a part of a Pipeline Ojbect

    Follows: https://spark.apache.org/docs/latest/ml-pipeline.html#transformers

    Arguments:
        Transformer {[object]} -- Inherited Classes
        HasInputCol {[object]} -- Inherited Classes
        HasOutputCol {[object]} -- Inherited Classes

    Returns:
        [pyspark DataFrame]
    """
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        super(IndReconstructer, self).__init__()
        # kwargs = self._input_kwargs
        # self.setParams(**kwargs)
        self.inputCol = inputCol
        self.outputCol = outputCol
    @keyword_only
    def setParams(self, inputCol=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        """transform method-- iterates through the oneHotDummy columns for a categorical variable and returns the index of the column that is closest to one. This corresponds to the stringIndexed value of this feature for this row.

        Arguments:
            dataset {[pyspark DataFrame]} -- dataset with OneHotDummy columns

        Returns:
            [pyspark DataFrame] -- dataset with column corresponding to a categorical indexed column
        """
        inColBase = self.inputCol
        outCol = self.outputCol

        closestToOne = F.udf(lambda x:abs(x-1), DoubleType())
        dummies = dataset.select(*[closestToOne(i).alias(i) if inColBase in i else i for i in dataset.columns if inColBase in i or i == 'SUBJECT'])
        dummies = dummies.withColumn('least_val', F.lit(F.least(*[F.col(i) for i in dataset.columns if inColBase in i])))

        dummies = dummies.select(*[(F.col(i) == F.col('least_val')).alias(i+'isind') if inColBase in i else i for i in dataset.columns if inColBase in i or i == 'SUBJECT'])
        getActive = F.udf(lambda row: [idx for idx, val in enumerate(row) if val][0], IntegerType())
        dummies = dummies.withColumn(outCol, getActive(F.struct(*[F.col(x) for x in dummies.columns if x != 'SUBJECT']).alias('struct')))
        dataset = dataset.join(dummies.select(['SUBJECT', outCol]), 'SUBJECT')

        return dataset

## Pipeline Functions
def get_string_pipeline(df, cols_to_exclude):
    """ Generates a list of preprocessing stages

    Arguments:
        df {[pyspark dataframe]} -- DataFrame including only the training data
        cols_to_exclude {[list]} -- Column names we don't want to to include in the preprocessing (i.e. SUBJECT/ target column)

    Returns:
        stages -- [{list}] list of pipeline stages to be used in preprocessing
        Continuous_Columns -- [{list}] list of columns that contain numeric features
    """

    String_Columns = []
    Continuous_Columns = []
    for _col, _type in df.dtypes: # This is a tuple of (<col name>, data type)
        if _col in cols_to_exclude:
            continue
        if _type == 'string':
            String_Columns.append(_col)
        elif _type == 'double' or _type == 'int' or _type == 'float':
            Continuous_Columns.append(_col)
        else:
            print("Unhandled Data type = {}".format((_col,_type)))
            continue

    # String Inexing
    str_indexers = [StringIndexer(inputCol=c, outputCol=c+'_ind', handleInvalid='skip') for c in String_Columns]
    indexed_string_vars = [c+'_ind' for c in String_Columns]
    # One hot encoding
    str_hot = [OneHotEncoder(inputCol = c+'_ind', outputCol = c+'_vec', dropLast=False)for c in String_Columns]
    encoded_str_vars = [c+'_vec' for c in String_Columns]
    # Converting the sparse vector to dummy columns
    str_dumbers = [OneHotDummies(inputCol = c+'_vec', outputCol = '_dummy') for c in String_Columns]
    # str_dumb_cols = [c for dummy in str_dumbers for c in dummy.getOutCols()]
    stages = str_indexers + str_hot + str_dumbers

    return stages, Continuous_Columns

def vector_assembler_pipeline(df, columns, doPCA = False, k =10):
    """After preprocessing String Columns, this function can be used to assemble a feature vector to be used for learning

    creates the following stages: VectorAssembler -> Standard Scalar [{ -> PCA}]

    Arguments:
        df {[pyspark DataFrame]} -- DataFrame containing preprocessed Columns
        columns {[list]} -- list of Column names of the preprocessed columns

    Keyword Arguments:
        doPCA {bool} -- Do you want to do PCA as part of the vector assembler (default: {False})
        k {int} -- Number of Principal Components to use (default: {10})

    Returns:
        [list] -- List of vector assembling stages
    """

    assembler = VectorAssembler(inputCols = columns, outputCol = 'featuresVec')
    scaler = StandardScaler(inputCol="featuresVec", outputCol="features", withStd=True, withMean=True) # centering and standardizing the data

    if doPCA:
        pca_obj = PCA(k=k, inputCol="features", outputCol="pcaFeatures")
        stages = [assembler, scaler, pca_obj]
    else:
        stages = [assembler, scaler]
    return stages

def postprocessing_pipeline(df, cols_to_exclude):
    """Assemble postprocessing pipeline to reconstruct original categorical indexed values from OneHotDummy Columns

    Arguments:
        df {[pyspark DataFrame]} -- DataFrame Including the original string Columns
        cols_to_exclude {[list]} -- list of columns to exclude

    Returns:
        reconstructers -- list of IndReconstructer stages
        String_Columns -- list of columns that are being reconstructed
    """
    String_Columns = []
    Continuous_Columns = []
    for _col, _type in df.dtypes: # This is a tuple of (<col name>, data type)
        if _col in cols_to_exclude:
            continue
        if _type == 'string':
            String_Columns.append(_col)
        elif _type == 'double' or _type == 'int' or _type == 'float':
            Continuous_Columns.append(_col)
        else:
            print("Unhandled Data type = {}".format((_col,_type)))
            continue

    # Extracting the Value of the OneHotEncoded Variable
    reconstructors = [IndReconstructer(inputCol = c, outputCol = c+'_activeInd') for c in String_Columns]
    return reconstructors, String_Columns

# Distribution fitting Functions
def make_pdf(dist, params, size=10000):
    """ """

    """Generate distributions's Probability Distribution Function

    Arguments:
        dist {[type]} -- [description]
        params {[type]} -- [description]

    Keyword Arguments:
        size {int} -- [description] (default: {10000})

    Returns:
        [type] -- [description]
    """
    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf

def best_fit_distribution(data, col_name, bins, ax):
    '''Model data by finding best fit distribution to data'''
    # Get histogram of original data

    output = dist_explore.pandas_histogram(data, bins=bins)
    output.reset_index(level=0, inplace=True)
    output['index'] = output['index'].apply(lambda x: np.mean([float(i.strip()) for i in x.split('-')]))
    output[col_name] = output[col_name]/np.sum(output[col_name])/(output['index'][1]-(output['index'][0]))

    x = output['index']
    y = output[col_name]
    # DISTRIBUTIONS = [
    #     st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
    #     st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
    #     st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
    #     st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
    #     st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
    #     st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,st.levy_stable,
    #     st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
    #     st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
    #     st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
    #     st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy
    # ]

    DISTRIBUTIONS = [
        st.beta,st.expon,
        st.halfnorm,
        st.norm,
        st.lognorm,
        st.uniform
    ]

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution in tqdm(DISTRIBUTIONS):

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data.collect())

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution

                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y.values - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        if sse < 0.05:
                            # Don't want to plot really bad ones
                            ax = pdf.plot(legend= True, label = distribution.name)
                            # ax.plot(x,pdf, label = distribution.name)
                            ax.legend()
                except Exception:
                    pass

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            pass

    return (best_distribution.name, best_params, best_sse)

## PCA Functions

def estimateCovariance(df, features_col = 'features'):
    """Compute the covariance matrix for a given dataframe.

    Note:
        The multi-dimensional covariance array should be calculated using outer products.  Don't
        forget to normalize the data by first subtracting the mean.

    Args:
        df:  A Spark dataframe
        features_col: name of the column with the features
    Returns:
        np.ndarray: A multi-dimensional array where the number of rows and columns both equal the
            length of the arrays in the input dataframe.
    """
    m = df.select(df[features_col]).rdd.map(lambda x: x[0]).mean()

    dfZeroMean = df.select(df[features_col]).rdd.map(lambda x:   x[0]).map(lambda x: x-m)  # subtract the mean

    return dfZeroMean.map(lambda x: np.outer(x,x)).sum()/df.count()

def mypca(df, k=10):
    """Computes the top `k` principal components, corresponding scores, and all eigenvalues.

    Note:
        All eigenvalues should be returned in sorted order (largest to smallest). `eigh` returns
        each eigenvectors as a column.  This function should also return eigenvectors as columns.

    Args:
        df: A Spark dataframe with a 'features' column, which (column) consists of DenseVectors.
        k (int): The number of principal components to return.

    Returns:
        tuple of (np.ndarray, RDD of np.ndarray, np.ndarray): A tuple of (eigenvectors, `RDD` of
        scores, eigenvalues).  Eigenvectors is a multi-dimensional array where the number of
        rows equals the length of the arrays in the input `RDD` and the number of columns equals
        `k`.  The `RDD` of scores has the same number of rows as `data` and consists of arrays
        of length `k`.  Eigenvalues is an array of length d (the number of features).
     """
    cov = estimateCovariance(df)
    col = cov.shape[1]
    eigVals, eigVecs = eigh(cov)
    inds = np.argsort(eigVals)
    eigVecs = eigVecs.T[inds[-1:-(col+1):-1]]
    components = eigVecs[0:k]
    eigVals = eigVals[inds[-1:-(col+1):-1]]  # sort eigenvals
    score = df.select(df['features']).rdd.map(lambda x: x[0]).map(lambda x: np.dot(x, components.T) )
    # Return the `k` principal components, `k` scores, and all eigenvalues

    return components.T, score, eigVals

def varianceExplained(df, k=10):
    components, scores, eigenvalues = mypca(df, k)
    return sum(eigenvalues[0:k])/sum(eigenvalues), components, scores, eigenvalues

# PCA reconstruction Functions

def reconstructPCA(sql, df, pc, mean, std, originalColumns, fits, pcaColumn = 'pcaFeatures'):

    """ Reconstruct data from lower dimensional space after performing PCA

    Arguments:
        sql {[pyspark.sql.context.SQLContext]}
        df {[spark dataframe]} -- inputted DataFrame
        pc {[numpy.ndarray]} -- principal components projected onto
        mean {[numpy.ndarray]} -- mean of original columns
        std {[numpy.ndarray]} -- std of original columns
        originalColumns {[list of strings]} -- original space column names
        fits {[dict]} -- fits of features returned from fitting routine

    Keyword Arguments:
        pcaColumn {str} -- column in df that contains PCA features (default: {'pcaFeatures'})

    Returns:
        [spark dataframe] -- dataframe containing reconstructed data
    """

    cols = df.columns
    cols.remove(pcaColumn)

    pddf = df.toPandas()
    first_series = pddf['pcaFeatures'].apply(lambda x : np.array(x.toArray())).as_matrix().reshape(-1,1)
    first_features = np.apply_along_axis(lambda x : x[0], 1, first_series)
    # undo-ing PCA
    first_reconstructed = np.dot(first_features, pc)
    # undo-ing the scaling
    first_reconstructed = np.multiply(first_reconstructed, std)+ mean
    first_reconstructedDF = pd.DataFrame(first_reconstructed, columns = originalColumns)
    for _col in cols:
        first_reconstructedDF[_col] = pddf[_col]

    # This is a pyspark Dataframe containing the reconstructed data, including the dummy columns for the string variables-- next step is to reverse the one-hot-encoding for the string columns
    first_reconstructed = sql.createDataFrame(first_reconstructedDF)

    cols_to_exclude = ['DATE_OF_STUDY']
    postPipeStages, String_Columns = postprocessing_pipeline(df, cols_to_exclude)

    postPipe = Pipeline(stages =postPipeStages)
    out = postPipe.fit(first_reconstructed).transform(first_reconstructed)
    for _col in String_Columns:
        out = out.join(df.select([_col, _col+'_ind'])\
                        .withColumnRenamed(_col+'_ind',_col+'_activeInd'), _col+'_activeInd')\
                        .dropDuplicates()
    cols_to_drop = [_col for _col in out.columns if any([base in _col for base in String_Columns]) and '_' in _col]

    reconstructedDF = out.drop(*cols_to_drop) # This is the equivalent as the first translated reconstructed dataframe above
    clip = F.udf(lambda x: x if x > 0 else 0.0, DoubleType())
    for _key in fits.keys():
        if fits[_key]['dist'] == 'EMPIRICAL':
            reconstructedDF = reconstructedDF.withColumn(_key,F.round(clip(F.col(_key))))
        else :
            reconstructedDF = reconstructedDF.withColumn(_key, clip(F.col(_key)))

    return reconstructedDF

## A simple Markov Chain object to help with generating next states for a patient
class MarkovChain(object):
    def __init__(self, transition_prob):
        """
        Initialize the MarkovChain instance.

        Parameters
        ----------
        transition_prob: dict
            A dict object representing the transition
            probabilities in Markov Chain.
            Should be of the form:
                {'state1': {'state1': 0.1, 'state2': 0.4},
                 'state2': {...}}
        """
        self.transition_prob = transition_prob
        self.states = list(transition_prob.keys()) # states that have transitions to the next layer
        # For states in the form <stateN_M> where N is the visit (layer) and M is the cluster in the N-th Layer
        self.max_num_steps = max([int(i.split('state')[1][0]) for i in self.states])

    def get_max_num_steps(self):
        return self.max_num_steps

    def next_state(self, current_state):
        """
        Returns the state of the random variable at the next time
        instance.

        Parameters
        ----------
        current_state: str
            The current state of the system.
        """
        try:

            # if not current_state in self.states:
            #     print('We have reached node {} where we do not know where they go from here... \n try reducing the number of clusters at level {} \n otherwise we might be at the terminating layer'.format(current_state, int(current_state.split('state')[1][0])))
            #     raise Exception('Unknown transition')

            next_possible_states = self.transition_prob[current_state].keys()
            return np.random.choice(
                next_possible_states,
                p=[self.transition_prob[current_state][next_state]
                for next_state in next_possible_states]
            )[:]
        except Exception as e:
            raise e

    def generate_states(self, current_state, no=10, last = True):
        """
        Generates the next states of the system.

        Parameters
        ----------
        current_state: str
            The state of the current random variable.
        no: int
            The number of future states to generate.
        last: bool
            Do we want to return just the last value
        """
        try:
            if no > self.max_num_steps:
                print('Number of steps exceeds the max number of possible next steps')
                raise Exception('<no> should not exceed {}. The value of <no> was: {}'.format(self.max_num_steps,no))

            future_states = []
            for i in range(no):
                try:
                    next_state = self.next_state(current_state)
                except Exception as e:
                    raise e
                future_states.append(next_state)
                current_state = next_state
            if last:
                return future_states[-1]
            else:
                return future_states
        except Exception as e:
            raise e

    def rep_states(self, current_state, no = 10, num_reps = 10):
        """running generate states a bunch of times and returning the final state that happens the most

        Arguments:
            current_state str -- The state of the current random variable
            no int -- number of time steps in the future to run
            num_reps int -- number of times to run the simultion forward

        Returns:
            state -- the most commonly reached state at the end of these runs
        """
        if no > self.max_num_steps:
            print('Number of steps exceeds the max number of possible next steps')
            raise Exception('<no> should not exceed {}. The value of <no> was: {}'.format(self.max_num_steps,no))

        endstates = []
        for _ in range(num_reps):
            endstates.append(self.generate_states(current_state, no = no, last= True))
        return max(set(endstates), key=endstates.count)

