import random
import warnings
from collections import OrderedDict, defaultdict
from multiprocessing.pool import ThreadPool

import graphviz
import numpy as np
import pandas as pd
import pyspark_dist_explore as dist_explore
import scipy.stats as st
from IPython.display import HTML
from numpy.linalg import eigh
from pyspark import keyword_only
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.evaluation import (BinaryClassificationEvaluator,
                                   MulticlassClassificationEvaluator,
                                   RegressionEvaluator)
from pyspark.ml.feature import (PCA, Bucketizer, OneHotEncoder, StandardScaler,
                                StringIndexer, VectorAssembler)
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param
from pyspark.ml.tuning import (CrossValidator, CrossValidatorModel,
                               ParamGridBuilder)
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import Row
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, DoubleType, IntegerType, StringType
from tqdm import tqdm


def get_confusion_matrix(spark, TP, TN, FP, FN):
    """
    Creates and returns a confusion matrix

    :param TP: True Positives
    :param TN: True Negatives
    :param FP: False Positives
    :param FN: False Negatives

    :return: Spark DataFrame
    """

    row = Row('', 'True', 'False')
    confusion_matrix = spark._wrapped.createDataFrame([row('True', TP, FN),
                                                       row('False', FP, TN)])
    return confusion_matrix


class SpliceBaseEvaluator(object):
    """
    Base ModelEvaluator
    """

    def __init__(self, spark, evaluator, supported_metrics, predictionCol="prediction",
                 labelCol="label"):
        """
        Constructor for SpliceBaseEvaluator
        
        :param spark: spark session
        :param evaluator: evaluator class from spark
        :param supported_metrics: supported metrics list
        :param predictionCol: prediction column
        :param labelCol: label column
        """
        self.spark = spark
        self.ev = evaluator
        self.prediction_col = predictionCol
        self.label = labelCol
        self.supported_metrics = supported_metrics
        self.avgs = defaultdict(list)

    def input(self, predictions_dataframe):
        """
        Input a dataframe
        
        :param ev: evaluator class
        :param predictions_dataframe: input df
        :return: none
        """
        for metric in self.supported_metrics:
            evaluator = self.ev(
                labelCol=self.label, predictionCol=self.prediction_col, metricName=metric)
            self.avgs[metric].append(evaluator.evaluate(predictions_dataframe))
            print("Current {metric}: {metric_val}".format(metric=metric,
                                                          metric_val=self.avgs
                                                          [metric][-1]))

    def get_results(self, as_dict=False):
        """
        Get Results
        
        :param dict: whether to get results in a dict or not
        :return: dictionary
        """
        computed_avgs = {}
        for key in self.avgs:
            computed_avgs[key] = np.mean(self.avgs[key])

        if as_dict:
            return computed_avgs

        metrics_row = Row(*self.supported_metrics)
        computed_row = metrics_row(*[float(computed_avgs[i]) for i in self.supported_metrics])
        return self.spark._wrapped.createDataFrame([computed_row])


class SpliceBinaryClassificationEvaluator(SpliceBaseEvaluator):
    """
    A Splice Machine evaluator for Spark Binary Classification models. Implements functions from SpliceBaseEvaluator.
    """
    def __init__(self, spark, predictionCol="prediction", labelCol="label", confusion_matrix=True):
        self.avg_tp = []
        self.avg_tn = []
        self.avg_fn = []
        self.avg_fp = []
        self.confusion_matrix = confusion_matrix

        supported = ["areaUnderROC", "areaUnderPR", 'TPR', 'SPC', 'PPV', 'NPV', 'FPR', 'FDR', 'FNR', 'ACC', 'F1', 'MCC']
        SpliceBaseEvaluator.__init__(self, spark, BinaryClassificationEvaluator, supported, predictionCol=predictionCol,
                                     labelCol=labelCol)

    def input(self, predictions_dataframe):
        """
        Evaluate actual vs Predicted in a dataframe
        
        :param predictions_dataframe: the dataframe containing the label and the predicition
        """
        for metric in self.supported_metrics:
            if metric in ['areaUnderROC', 'areaUnderPR']:
                evaluator = self.ev(labelCol=self.label, rawPredictionCol=self.prediction_col, metricName=metric)

                self.avgs[metric].append(evaluator.evaluate(predictions_dataframe))
                print("Current {metric}: {metric_val}".format(metric=metric,
                                                              metric_val=self.avgs
                                                              [metric][-1]))

        pred_v_lab = predictions_dataframe.select(self.label,
                                                  self.prediction_col)  # Select the actual and the predicted labels

        # Add confusion stats
        self.avg_tp.append(pred_v_lab[(pred_v_lab[self.label] == 1)
                                      & (pred_v_lab[self.prediction_col] == 1)].count())
        self.avg_tn.append(pred_v_lab[(pred_v_lab[self.label] == 0)
                                      & (pred_v_lab[self.prediction_col] == 0)].count())
        self.avg_fp.append(pred_v_lab[(pred_v_lab[self.label] == 1)
                                      & (pred_v_lab[self.prediction_col] == 0)].count())
        self.avg_fn.append(pred_v_lab[(pred_v_lab[self.label] == 0)
                                      & (pred_v_lab[self.prediction_col] == 1)].count())

        TP = np.mean(self.avg_tp)
        TN = np.mean(self.avg_tn)
        FP = np.mean(self.avg_fp)
        FN = np.mean(self.avg_fn)

        self.avgs['TPR'].append(float(TP) / (TP + FN))
        self.avgs['SPC'].append(float(TP) / (TP + FN))
        self.avgs['TNR'].append(float(TN) / (TN + FP))
        self.avgs['PPV'].append(float(TP) / (TP + FP))
        self.avgs['NPV'].append(float(TN) / (TN + FN))
        self.avgs['FNR'].append(float(FN) / (FN + TP))
        self.avgs['FPR'].append(float(FP) / (FP + TN))
        self.avgs['FDR'].append(float(FP) / (FP + TP))
        self.avgs['FOR'].append(float(FN) / (FN + TN))
        self.avgs['ACC'].append(float(TP + TN) / (TP + FN + FP + TN))
        self.avgs['F1'].append(float(2 * TP) / (2 * TP + FP + FN))
        self.avgs['MCC'].append(float(TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))

        if self.confusion_matrix:
            get_confusion_matrix(
                self.spark,
                float(TP),
                float(TN),
                float(FP),
                float(FN)
            ).show()

    def plotROC(self, fittedEstimator, ax):
        """
        Plots the receiver operating characteristic curve for the trained classifier
        
        :param fittedEstimator: fitted logistic regression model
        :param ax: matplotlib axis object
        :return: axis with ROC plot
        """
        if fittedEstimator.__class__ == LogisticRegressionModel:
            trainingSummary = fittedEstimator.summary
            roc = trainingSummary.roc.toPandas()
            ax.plot(roc['FPR'], roc['TPR'], label='Training set areaUnderROC: \n' + str(trainingSummary.areaUnderROC))
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend()
            return ax
        else:
            raise NotImplementedError("Only supported for Logistic Regression Models")


class SpliceRegressionEvaluator(SpliceBaseEvaluator):
    """
    A Splice Machine evaluator for Spark Regression models. Implements functions from SpliceBaseEvaluator.
    """

    def __init__(self, spark, predictionCol="prediction", labelCol="label"):
        supported = ['rmse', 'mse', 'r2', 'mae']
        SpliceBaseEvaluator.__init__(self, spark, RegressionEvaluator, supported, predictionCol=predictionCol,
                                     labelCol=labelCol)


class SpliceMultiClassificationEvaluator(SpliceBaseEvaluator):
    """
    A Splice Machine evaluator for Spark MultiClass models. Implements functions from SpliceBaseEvaluator.
    """

    def __init__(self, spark, predictionCol="prediction", labelCol="label"):
        supported = ["f1", "weightedPrecision", "weightedRecall", "accuracy"]
        SpliceBaseEvaluator.__init__(self, spark, MulticlassClassificationEvaluator, supported,
                                     predictionCol=predictionCol, labelCol=labelCol)


class DecisionTreeVisualizer(object):
    """
    Visualize a decision tree, either in code like format, or graphviz
    """

    @staticmethod
    def feature_importance(spark, model, dataset, featuresCol="features"):
        """
        Return a dataframe containing the relative importance of each feature
        
        :param model: The Spark Machine Learning model
        :param dataframe: Spark Dataframe
        :param featureCol: (str) the column containing the feature vector
        :return: dataframe containing importance
        """
        import pandas as pd
        featureImp = model.featureImportances
        list_extract = []
        for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
            list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][
                i]
        varlist = pd.DataFrame(list_extract)
        varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
        return spark._wrapped.createDataFrame((varlist.sort_values('score', ascending=False)))

    @staticmethod
    def visualize(
            model,
            feature_column_names,
            label_names,
            size=None,
            horizontal=False,
            tree_name='tree',
            visual=False,
    ):
        """
        Visualize a decision tree, either in a code like format, or graphviz
        
        :param model: the fitted decision tree classifier\n
        :param feature_column_names: (List[str]) column names for features
            You can access these feature names by using your VectorAssembler (in PySpark) and calling it's .getInputCols() function
        :param label_names: (List[str]) labels vector (below avg, above avg)
        :param size: tuple(int,int) The size of the graph. If unspecified, graphviz will automatically assign a size
        :param horizontal: (Bool) if the tree should be rendered horizontally
        :param tree_name: the name you would like to call the tree
        :param visual: bool, true if you want a graphviz pdf containing your file
        :return dot: The graphvis object
        """

        tree_to_json = DecisionTreeVisualizer.replacer(model.toDebugString,
                                                       ['feature ' + str(i) for i in
                                                        range(len(feature_column_names) - 1, -1, -1)],
                                                       reversed(feature_column_names))

        tree_to_json = DecisionTreeVisualizer.replacer(tree_to_json,
                                                       [f'Predict: {str(i)}.0' for i in
                                                        range(len(label_names) - 1, -1, -1)],
                                                       reversed(label_names))
        if not visual:
            return tree_to_json

        dot = graphviz.Digraph(comment='Decision Tree')
        if size:
            dot.attr(size=size)
        if horizontal:
            dot.attr(rankdir="LR")
        dot.node_attr.update(color='lightblue2', style='filled')
        json_d = DecisionTreeVisualizer.tree_json(tree_to_json)

        DecisionTreeVisualizer.add_node(dot, '', '', json_d,
                                        realroot=True)
        dot.render(tree_name)
        print(f'Generated pdf file of tree. You can view it in your Jupyter directory under {dot.filepath}.pdf\n')
        dot.view()
        return (dot)

    @staticmethod
    def replacer(string, bad, good):
        """
        Replace every string in "bad" with the corresponding string in "good"
        
        :param string: string to replace in
        :param bad: array of strings to replace
        :param good: array of strings to replace with
        :return:
        """

        for b, g in zip(bad, good):
            string = string.replace(b, g)
        return string

    @staticmethod
    def add_node(
            dot,
            parent,
            node_hash,
            root,
            realroot=False,
    ):
        """
        Traverse through the .debugString json and generate a graphviz tree
        
        :param dot: dot file object
        :param parent: not used currently
        :param node_hash: unique node id
        :param root: the root of tree
        :param realroot: whether or not it is the real root, or a recursive root
        :return:
        """

        node_id = str(hash(root['name'])) + str(random.randint(0, 100))
        if root:
            dot.node(node_id, root['name'])
            if not realroot:
                dot.edge(node_hash, node_id)
            if root.get('children'):
                if not root['children'][0].get('children'):
                    DecisionTreeVisualizer.add_node(dot, root['name'],
                                                    node_id, root['children'][0])
                else:
                    DecisionTreeVisualizer.add_node(dot, root['name'],
                                                    node_id, root['children'][0])
                    DecisionTreeVisualizer.add_node(dot, root['name'],
                                                    node_id, root['children'][1])

    @staticmethod
    def parse(lines):
        """
        Lines in debug string
       
        :param lines:
        :return: block json
        """

        block = []
        while lines:
            if lines[0].startswith('If'):
                bl = ' '.join(lines.pop(0).split()[1:]).replace('(', ''
                                                                ).replace(')', '')
                block.append({'name': bl,
                              'children': DecisionTreeVisualizer.parse(lines)})

                if lines[0].startswith('Else'):
                    be = ' '.join(lines.pop(0).split()[1:]).replace('('
                                                                    , '').replace(')', '')
                    block.append({'name': be,
                                  'children': DecisionTreeVisualizer.parse(lines)})
            elif not lines[0].startswith(('If', 'Else')):
                block2 = lines.pop(0)
                block.append({'name': block2})
            else:
                break
        return block

    @staticmethod
    def tree_json(tree):
        """
        Generate a JSON representation of a decision tree
        
        :param tree: tree debug string
        :return: json
        """

        data = []
        for line in tree.splitlines():
            if line.strip():
                line = line.strip()
                data.append(line)
            else:
                break
            if not line:
                break
        res = [{'name': 'Root',
                'children': DecisionTreeVisualizer.parse(data[1:])}]
        return res[0]


def inspectTable(spliceMLCtx, sql, topN=5):
    """
    Inspect the values of the columns of the table (dataframe) returned from the sql query
    
    :param spliceMLCtx: SpliceMLContext
    :param sql: sql string to execute
    :param topN: the number of most frequent elements of a column to return, defaults to 5
    """
    df = spliceMLCtx.df(sql)
    df = df.repartition(50)

    for _col, _type in df.dtypes:
        print("------Inspecting column {} -------- ".format(_col))

        val_counts = df.groupby(_col).count()
        val_counts.show()
        val_counts.orderBy(F.desc('count')).limit(topN).show()

        if _type == 'double' or _type == 'int':
            df.select(_col).describe().show()



# Custom Transformers
class Rounder(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable):
    """Transformer to round predictions for ordinal regression
    Follows: https://spark.apache.org/docs/latest/ml-pipeline.html#transformers
    
    :param Transformer: Inherited Class
    :param HasInputCol: Inherited Class
    :param HasOutputCol: Inherited Class
    :return: Transformed Dataframe with rounded predictionCol
    
    :Example:
        .. code-block:: python

            >>> from pyspark.sql.session import SparkSession
            >>> from splicemachine.stats.stats import Rounder
            >>> spark = SparkSession.builder.getOrCreate()
            >>> dataset = spark.createDataFrame(
            ...      [(0.2, 0.0),
            ...       (1.2, 1.0),
            ...       (1.6, 2.0),
            ...       (1.1, 0.0),
            ...       (3.1, 0.0)],
            ...      ["prediction", "label"])
            >>> dataset.show()
            +----------+-----+
            |prediction|label|
            +----------+-----+
            |       0.2|  0.0|
            |       1.2|  1.0|
            |       1.6|  2.0|
            |       1.1|  0.0|
            |       3.1|  0.0|
            +----------+-----+
            >>> rounder = Rounder(predictionCol = "prediction", labelCol = "label",
                clipPreds = True)
            >>> rounder.transform(dataset).show()
            +----------+-----+
            |prediction|label|
            +----------+-----+
            |       0.0|  0.0|
            |       1.0|  1.0|
            |       2.0|  2.0|
            |       1.0|  0.0|
            |       2.0|  0.0|
            +----------+-----+
            >>> rounderNoClip = Rounder(predictionCol = "prediction", labelCol = "label",
                clipPreds = False)
            >>> rounderNoClip.transform(dataset).show()
            +----------+-----+
            |prediction|label|
            +----------+-----+
            |       0.0|  0.0|
            |       1.0|  1.0|
            |       2.0|  2.0|
            |       1.0|  0.0|
            |       3.0|  0.0|
            +----------+-----+
    """

    @keyword_only
    def __init__(self, predictionCol="prediction", labelCol="label", clipPreds=True, maxLabel=None, minLabel=None):
        """
        initialize self

        :param predictionCol: column containing predictions, defaults to "prediction"
        :param labelCol: column containing labels, defaults to "label"
        :param clipPreds: clip all predictions above a specified maximum value
        :param maxLabel: optional: the maximum value for the prediction column, otherwise uses the maximum of the labelCol, defaults to None
        :param minLabel: optional: the minimum value for the prediction column, otherwise uses the maximum of the labelCol, defaults to None
        """
        super(Rounder, self).__init__()
        self.labelCol = labelCol
        self.predictionCol = predictionCol
        self.clipPreds = clipPreds
        self.maxLabel = maxLabel
        self.minLabel = minLabel

    @keyword_only
    def setParams(self, predictionCol="prediction", labelCol="label"):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        """
        Rounds the predictions to the nearest integer value, and also clips them at the max/min value observed in label
        
        :param dataset: dataframe with predictions to be rounded
        :return: DataFrame with rounded predictions
        """
        labelCol = self.labelCol
        predictionCol = self.predictionCol

        if self.clipPreds:
            max_label = self.maxLabel if self.maxLabel else dataset.agg({labelCol: 'max'}).collect()[0][0]
            min_label = self.minLabel if self.minLabel else dataset.agg({labelCol: 'min'}).collect()[0][0]
            clip = F.udf(lambda x: float(max_label) if x > max_label else (float(min_label) if x < min_label else x),
                         DoubleType())

            dataset = dataset.withColumn(predictionCol, F.round(clip(F.col(predictionCol))))
        else:
            dataset = dataset.withColumn(predictionCol, F.round(F.col(predictionCol)))

        return dataset


class OneHotDummies(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable):
    """
    Transformer to generate dummy columns for categorical variables as a part of a preprocessing pipeline\n
    Follows: https://spark.apache.org/docs/latest/ml-pipeline.html#transformers
    
    :param Transformer: Inherited Classes
    :param HasInputCol: Inherited Classes
    :param HasOutputCol: Inherited Classes
    :return: pyspark DataFrame
    """

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        """
        Assigns variables to parameters passed

        :param inputCol: Sparse vector returned by OneHotEncoders, defaults to None
        :param outputCol: string base to append to output columns names, defaults to None
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
        """
        Iterates through the number of categorical values of a categorical variable and adds dummy columns for each of those categories
        For a string categorical column, include this transformer in the following workflow: StringIndexer -> OneHotEncoder -> OneHotDummies -> PCA/ Learning Algorithm

        :param dataset: PySpark DataFrame where inputCol is the column  returned by by OneHotEncoders
        :return: original DataFrame with M additional columns where M = # of categories for this variable
        """
        out_col_suffix = self.outputCol  # this is what I want to append to the column name
        col_name = self.inputCol

        out_col_base = col_name + out_col_suffix  # this is the base for the n outputted columns

        # helper functions
        get_num_categories = F.udf(lambda x: int(x.size), IntegerType())
        get_active_index = F.udf(lambda x: int(x.indices[0]), IntegerType())
        check_active_index = F.udf(lambda active, i: int(active == i), IntegerType())

        num_categories = dataset.select(
            get_num_categories(col_name).alias('num_categories')).distinct()  # this returns a dataframe
        if num_categories.count() == 1:  # making sure all the sparse vectors have the same number of categories
            num_categories_int = num_categories.collect()[0]['num_categories']  # now this is an int

        dataset = dataset.withColumn('active_index', get_active_index(col_name))
        column_names = []
        for i in range(num_categories_int):  # Now I'm going to make a column for each category
            column_name = out_col_base + '_' + str(i)
            dataset = dataset.withColumn(column_name, check_active_index(F.col('active_index'), F.lit(i)))
            column_names.append(column_name)

        dataset = dataset.drop('active_index')
        self.outcols = column_names
        return dataset

    def getOutCols(self):
        return self.outcols


class IndReconstructer(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable):
    """
    Transformer to reconstruct String Index from OneHotDummy Columns. This can be used as a part of a Pipeline Ojbect\n
    Follows: https://spark.apache.org/docs/latest/ml-pipeline.html#transformers

    :param Transformer: Inherited Class
    :param HasInputCol: Inherited Class
    :param HasOutputCol: Inherited Class
    :return: Transformed PySpark Dataframe With Original String Indexed Variables
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
        """
        Iterates through the oneHotDummy columns for a categorical variable and returns the index of the column that is closest to one. This corresponds to the stringIndexed value of this feature for this row.

        :param dataset: dataset with OneHotDummy columns
        :return: DataFrame with column corresponding to a categorical indexed column
        """
        inColBase = self.inputCol
        outCol = self.outputCol

        closestToOne = F.udf(lambda x: abs(x - 1), DoubleType())
        dummies = dataset.select(*[closestToOne(i).alias(i) if inColBase in i else i for i in dataset.columns if
                                   inColBase in i or i == 'SUBJECT'])
        dummies = dummies.withColumn('least_val',
                                     F.lit(F.least(*[F.col(i) for i in dataset.columns if inColBase in i])))

        dummies = dummies.select(
            *[(F.col(i) == F.col('least_val')).alias(i + 'isind') if inColBase in i else i for i in dataset.columns if
              inColBase in i or i == 'SUBJECT'])
        getActive = F.udf(lambda row: [idx for idx, val in enumerate(row) if val][0], IntegerType())
        dummies = dummies.withColumn(outCol, getActive(
            F.struct(*[F.col(x) for x in dummies.columns if x != 'SUBJECT']).alias('struct')))
        dataset = dataset.join(dummies.select(['SUBJECT', outCol]), 'SUBJECT')

        return dataset


class OverSampler(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable):
    """
    Transformer to oversample datapoints with minority labels\n
    
    Follows: https://spark.apache.org/docs/latest/ml-pipeline.html#transformers
    
    :param Transformer: Inherited Class
    :param HasInputCol: Inherited Class
    :param HasOutputCol: Inherited Class
    :return: PySpark Dataframe with labels in approximately equal ratios
    
    :Example:
        .. code-block:: python

            >>> from pyspark.sql import functions as F
            >>> from pyspark.sql.session import SparkSession
            >>> from pyspark.stats.linalg import Vectors
            >>> from splicemachine.stats.stats import OverSampler
            >>> spark = SparkSession.builder.getOrCreate()
            >>> df = spark.createDataFrame(
            ...      [(Vectors.dense([0.0]), 0.0),
            ...       (Vectors.dense([0.5]), 0.0),
            ...       (Vectors.dense([0.4]), 1.0),
            ...       (Vectors.dense([0.6]), 1.0),
            ...       (Vectors.dense([1.0]), 1.0)] * 10,
            ...      ["features", "Class"])
            >>> df.groupBy(F.col("Class")).count().orderBy("count").show()
            +-----+-----+
            |Class|count|
            +-----+-----+
            |  0.0|   20|
            |  1.0|   30|
            +-----+-----+
            >>> oversampler = OverSampler(labelCol = "Class", strategy = "auto")
            >>> oversampler.transform(df).groupBy("Class").count().show()
            +-----+-----+
            |Class|count|
            +-----+-----+
            |  0.0|   29|
            |  1.0|   30|
            +-----+-----+
    """

    @keyword_only
    def __init__(self, labelCol=None, strategy="auto", randomState=None):
        """
        Initialize self

        :param labelCol: Label Column name, defaults to None
        :param strategy: defaults to "auto", strategy to resample the dataset:
                        • Only currently supported for "auto" Corresponds to random samples with repleaement
        :param randomState: sets the seed of sample algorithm
        """
        super(OverSampler, self).__init__()
        self.labelCol = labelCol
        self.strategy = strategy
        self.withReplacement = True if strategy == "auto" else False
        self.randomState = np.random.randn() if not randomState else randomState

    @keyword_only
    def setParams(self, labelCol=None, strategy="auto"):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        """
        Oversamples

        :param dataset: dataframe to be oversampled
        :return: DataFrame with the resampled data points
        """
        if self.strategy == "auto":

            pd_value_counts = dataset.groupBy(F.col(self.labelCol)).count().toPandas()

            label_type = dataset.schema[self.labelCol].dataType.simpleString()
            types_dic = {'int': int, "string": str, "double": float}

            maxidx = pd_value_counts['count'].idxmax()

            self.majorityLabel = types_dic[label_type](pd_value_counts[self.labelCol].loc[maxidx])
            majorityData = dataset.filter(F.col(self.labelCol) == self.majorityLabel)

            returnData = None

            if len(pd_value_counts) == 1:
                raise ValueError(
                    f'Error! Number of labels = {len(pd_value_counts)}. Cannot Oversample with this number of classes')
            elif len(pd_value_counts) == 2:
                minidx = pd_value_counts['count'].idxmin()
                minorityLabel = types_dic[label_type](pd_value_counts[self.labelCol].loc[minidx])
                ratio = pd_value_counts['count'].loc[maxidx] / pd_value_counts['count'].loc[minidx] * 1.0

                returnData = majorityData.union(
                    dataset.filter(F.col(self.labelCol) == minorityLabel).sample(withReplacement=self.withReplacement,
                                                                                 fraction=ratio, seed=self.randomState))

            else:
                minority_labels = list(pd_value_counts.drop(maxidx)[self.labelCol])

                ratios = {types_dic[label_type](minority_label): pd_value_counts['count'].loc[maxidx] / float(
                    pd_value_counts[pd_value_counts[self.labelCol] == minority_label]['count']) for minority_label in
                          minority_labels}

                for (minorityLabel, ratio) in ratios.items():
                    minorityData = dataset.filter(F.col(self.labelCol) == minorityLabel).sample(
                        withReplacement=self.withReplacement, fraction=ratio, seed=self.randomState)
                    if not returnData:
                        returnData = majorityData.union(minorityData)
                    else:
                        returnData = returnData.union(minorityData)

            return returnData
        else:
            raise NotImplementedError("Only auto is currently implemented")


class OverSampleCrossValidator(CrossValidator):
    """
    Class to perform Cross Validation model evaluation while over-sampling minority labels.
    
    :Example:
        .. code-block:: python
            
            >>> from pyspark.sql.session import SparkSession
            >>> from pyspark.stats.classification import LogisticRegression
            >>> from pyspark.stats.evaluation import BinaryClassificationEvaluator,\n
             MulticlassClassificationEvaluator
            >>> from pyspark.stats.linalg import Vectors
            >>> from splicemachine.stats.stats import OverSampleCrossValidator
            >>> spark = SparkSession.builder.getOrCreate()
            >>> dataset = spark.createDataFrame(
            ...      [(Vectors.dense([0.0]), 0.0),
            ...       (Vectors.dense([0.5]), 0.0),
            ...       (Vectors.dense([0.4]), 1.0),
            ...       (Vectors.dense([0.6]), 1.0),
            ...       (Vectors.dense([1.0]), 1.0)] * 10,
            ...      ["features", "label"])
            >>> lr = LogisticRegression()
            >>> grid = ParamGridBuilder().addGrid(lr.maxIter, [0, 1]).build()
            >>> PRevaluator = BinaryClassificationEvaluator(metricName = 'areaUnderPR')
            >>> AUCevaluator = BinaryClassificationEvaluator(metricName = 'areaUnderROC')
            >>> ACCevaluator = MulticlassClassificationEvaluator(metricName="accuracy")
            >>> cv = OverSampleCrossValidator(estimator=lr, estimatorParamMaps=grid,
                    evaluator=AUCevaluator, altEvaluators = [PRevaluator, ACCevaluator],
                    parallelism=2,seed = 1234)
            >>> cvModel = cv.fit(dataset)
            >>> print(cvModel.avgMetrics)
            [(0.5, [0.5888888888888888, 0.3888888888888889]), (0.806878306878307,
                [0.8556863149300125, 0.7055555555555556])]
            >>> print(AUCevaluator.evaluate(cvModel.transform(dataset)))
            0.8333333333333333
            """

    def __init__(self, estimator, estimatorParamMaps, evaluator, numFolds=3, seed=None, parallelism=3,
                 collectSubModels=False, labelCol='label', altEvaluators=None, overSample=True):
        """ 
        Initialize Self

        :param estimator: Machine Learning Model, defaults to None
        :param estimatorParamMaps: paramMap to search, defaults to None
        :param evaluator: primary model evaluation metric, defaults to None
        :param numFolds: number of folds to perform, defaults to 3
        :param seed: random state, defaults to None
        :param parallelism: number of threads, defaults to 1
        :param collectSubModels: to return submodels, defaults to False
        :param labelCol: target variable column label, defaults to 'label'
        :param altEvaluators: additional metrics to evaluate, defaults to None
                             If passed, the metrics of the alternate evaluators are accessed in the CrossValidatorModel.avgMetrics attribute
        :param overSample: Boolean: to perform oversampling of minority labels, defaults to True
        """
        self.label = labelCol
        self.altEvaluators = altEvaluators
        self.toOverSample = overSample
        super(OverSampleCrossValidator, self).__init__(estimator=estimator, estimatorParamMaps=estimatorParamMaps,
                                                       evaluator=evaluator, numFolds=numFolds, seed=seed,
                                                       parallelism=parallelism, collectSubModels=collectSubModels)

    def getLabel(self):
        return self.label

    def getOversample(self):
        return self.toOverSample

    def getAltEvaluators(self):
        return self.altEvaluators

    def _parallelFitTasks(self, est, train, eva, validation, epm, collectSubModel, altEvaluators):
        """
        Creates a list of callables which can be called from different threads to fit and evaluate
        an estimator in parallel. Each callable returns an `(index, metric)` pair if altEvaluators, (index, metric, [alt_metrics]).
        
        :param est: Estimator, the estimator to be fit.
        :param train: DataFrame, training data set, used for fitting.
        :param eva: Evaluator, used to compute `metric`
        :param validation: DataFrame, validation data set, used for evaluation.
        :param epm: Sequence of ParamMap, params maps to be used during fitting & evaluation.
        :param collectSubModel: Whether to collect sub model.
        :return: (int, float, subModel), an index into `epm` and the associated metric value.
        """
        modelIter = est.fitMultiple(train, epm)

        def singleTask():
            index, model = next(modelIter)
            metric = eva.evaluate(model.transform(validation, epm[index]))
            altmetrics = None
            if altEvaluators:
                altmetrics = [altEva.evaluate(model.transform(validation, epm[index])) for altEva in altEvaluators]
            return index, metric, altmetrics, model if collectSubModel else None

        return [singleTask] * len(epm)

    def _fit(self, dataset):
        """
        Performs k-fold crossvaldidation on simple oversampled dataset
        
        :param dataset: full dataset
        :return: CrossValidatorModel containing the fitted BestModel with the average of the primary and alternate metrics in a list of tuples in the format: [(paramComb1_average_primary_metric, [paramComb1_average_altmetric1,paramComb1_average_altmetric2]), (paramComb2_average_primary_metric, [paramComb2_average_altmetric1,paramComb2_average_altmetric2])]
        """
        est = self.getOrDefault(self.estimator)
        epm = self.getOrDefault(self.estimatorParamMaps)
        numModels = len(epm)
        eva = self.getOrDefault(self.evaluator)
        nFolds = self.getOrDefault(self.numFolds)
        seed = self.getOrDefault(self.seed)

        # Getting Label and altEvaluators
        label = self.getLabel()
        altEvaluators = self.getAltEvaluators()
        altMetrics = [[0.0] * len(altEvaluators)] * numModels if altEvaluators else None
        h = 1.0 / nFolds
        randCol = self.uid + "_rand"
        df = dataset.select("*", F.rand(seed).alias(randCol))
        metrics = [0.0] * numModels

        pool = ThreadPool(processes=min(self.getParallelism(), numModels))
        subModels = None
        collectSubModelsParam = self.getCollectSubModels()
        if collectSubModelsParam:
            subModels = [[None for j in range(numModels)] for i in range(nFolds)]

        for i in range(nFolds):
            # Getting the splits such that no data is reused
            validateLB = i * h
            validateUB = (i + 1) * h
            condition = (df[randCol] >= validateLB) & (df[randCol] < validateUB)
            validation = df.filter(condition).cache()
            train = df.filter(~condition).cache()

            # Oversampling the minority class(s) here
            if self.toOverSample:
                withReplacement = True
                oversampler = OverSampler(labelCol=self.label, strategy="auto")

                # Oversampling
                train = oversampler.transform(train)
            # Getting the individual tasks so this can be parallelized
            tasks = self._parallelFitTasks(est, train, eva, validation, epm, collectSubModelsParam, altEvaluators)
            # Calling the parallel process
            for j, metric, fold_alt_metrics, subModel in pool.imap_unordered(lambda f: f(), tasks):
                metrics[j] += (metric / nFolds)
                if fold_alt_metrics:
                    altMetrics[j] = [altMetrics[j][i] + fold_alt_metrics[i] / nFolds for i in range(len(altEvaluators))]

                if collectSubModelsParam:
                    subModels[i][j] = subModel

            validation.unpersist()
            train.unpersist()

        if eva.isLargerBetter():
            bestIndex = np.argmax(metrics)
        else:
            bestIndex = np.argmin(metrics)
        bestModel = est.fit(dataset, epm[bestIndex])
        metrics = [(metric, altMetrics[idx]) for idx, metric in enumerate(metrics)]
        return self._copyValues(CrossValidatorModel(bestModel, metrics, subModels))


## Pipeline Functions
def get_string_pipeline(df, cols_to_exclude, steps=['StringIndexer', 'OneHotEncoder', 'OneHotDummies']):
    """
    Generates a list of preprocessing stages
    
    :param df: DataFrame including only the training data
    :param cols_to_exclude: Column names we don't want to to include in the preprocessing (i.e. SUBJECT/ target column)
    :param stages: preprocessing steps to take
    :return:  (stages, Numeric_Columns)
        stages: list of pipeline stages to be used in preprocessing
        Numeric_Columns: list of columns that contain numeric features
    """

    String_Columns = []
    Numeric_Columns = []
    for _col, _type in df.dtypes:  # This is a tuple of (<col name>, data type)
        if _col in cols_to_exclude:
            continue
        if _type == 'string':
            String_Columns.append(_col)
        elif _type == 'double' or _type == 'int' or _type == 'float':
            Numeric_Columns.append(_col)
        else:
            print("Unhandled Data type = {}".format((_col, _type)))
            continue

    stages = []
    if 'StringIndexer' in steps:
        # String Inexing
        str_indexers = [StringIndexer(inputCol=c, outputCol=c + '_ind', handleInvalid='skip') for c in String_Columns]
        indexed_string_vars = [c + '_ind' for c in String_Columns]
        stages = stages + str_indexers

    if 'OneHotEncoder' in steps:
        # One hot encoding
        str_hot = [OneHotEncoder(inputCol=c + '_ind', outputCol=c + '_vec', dropLast=False) for c in String_Columns]
        encoded_str_vars = [c + '_vec' for c in String_Columns]
        stages = stages + str_hot

    if 'OneHotDummies' in steps:
        # Converting the sparse vector to dummy columns
        str_dumbers = [OneHotDummies(inputCol=c + '_vec', outputCol='_dummy') for c in String_Columns]
        str_dumb_cols = [c for dummy in str_dumbers for c in dummy.getOutCols()]
        stages = stages + str_dumbers

    if len(stages) == 0:
        ERROR = """
        Parameter <steps> must include 'StringIndexer', 'OneHotEncoder', 'OneHotDummies'
        """
        print(ERROR)
        raise Exception(ERROR)

    return stages, Numeric_Columns


def vector_assembler_pipeline(df, columns, doPCA=False, k=10):
    """
    After preprocessing String Columns, this function can be used to assemble a feature vector to be used for learning
    creates the following stages: VectorAssembler -> Standard Scalar [{ -> PCA}]
    
    :param df: DataFrame containing preprocessed Columns
    :param columns: list of Column names of the preprocessed columns
    :param doPCA:  Do you want to do PCA as part of the vector assembler? defaults to False
    :param k:  Number of Principal Components to use, defaults to 10
    :return: List of vector assembling stages
    """

    assembler = VectorAssembler(inputCols=columns, outputCol='featuresVec')
    scaler = StandardScaler(inputCol="featuresVec", outputCol="features", withStd=True,
                            withMean=True)  # centering and standardizing the data

    if doPCA:
        pca_obj = PCA(k=k, inputCol="features", outputCol="pcaFeatures")
        stages = [assembler, scaler, pca_obj]
    else:
        stages = [assembler, scaler]
    return stages


def postprocessing_pipeline(df, cols_to_exclude):
    """
    Assemble postprocessing pipeline to reconstruct original categorical indexed values from OneHotDummy Columns
    
    :param df: DataFrame Including the original string Columns
    :param cols_to_exclude: list of columns to exclude
    :return: (reconstructers, String_Columns)
        reconstructers: list of IndReconstructer stages
        String_Columns: list of columns that are being reconstructed
    """
    String_Columns = []
    Numeric_Columns = []
    for _col, _type in df.dtypes:  # This is a tuple of (<col name>, data type)
        if _col in cols_to_exclude:
            continue
        if _type == 'string':
            String_Columns.append(_col)
        elif _type == 'double' or _type == 'int' or _type == 'float':
            Numeric_Columns.append(_col)
        else:
            print("Unhandled Data type = {}".format((_col, _type)))
            continue

    # Extracting the Value of the OneHotEncoded Variable
    reconstructors = [IndReconstructer(inputCol=c, outputCol=c + '_activeInd') for c in String_Columns]
    return reconstructors, String_Columns


# Distribution fitting Functions
def make_pdf(dist, params, size=10000):
    """
    Generate distributions's Probability Distribution Function
    
    :param dist: scipy.stats distribution object: https://docs.scipy.org/doc/scipy/reference/stats.html
    :param params: distribution parameters
    :param size: how many data points to generate , defaults to 10000
    :return: series of probability density function for this distribution
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
    """
    Model data by finding best fit distribution to data
    
    :param data: DataFrame with one column containing the feature whose distribution is to be investigated
    :param col_name: column name for feature
    :param bins: number of bins to use in generating the histogram of this data
    :param ax: axis to plot histogram on
    :return: (best_distribution.name, best_params, best_sse)
        best_distribution.name: string of the best distribution name
        best_params: parameters for this distribution
        best_sse: sum of squared errors for this distribution against the empirical pdf
    """
    # Get histogram of original data

    output = dist_explore.pandas_histogram(data, bins=bins)
    output.reset_index(level=0, inplace=True)
    output['index'] = output['index'].apply(lambda x: np.mean([float(i.strip()) for i in x.split('-')]))
    output[col_name] = output[col_name] / np.sum(output[col_name]) / (output['index'][1] - (output['index'][0]))

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
        st.beta, st.expon,
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
                            ax = pdf.plot(legend=True, label=distribution.name)
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

def estimateCovariance(df, features_col='features'):
    """
    Compute the covariance matrix for a given dataframe.
        
    
    :param df: PySpark dataframe
    :param features_col: name of the column with the features, defaults to 'features'
    :return: np.ndarray: A multi-dimensional array where the number of rows and columns both equal the length of the arrays in the input dataframe.

    :Note: 
        The multi-dimensional covariance array should be calculated using outer products.  Don't forget to normalize the data by first subtracting the mean.
    """
    m = df.select(df[features_col]).rdd.map(lambda x: x[0]).mean()

    dfZeroMean = df.select(df[features_col]).rdd.map(lambda x: x[0]).map(lambda x: x - m)  # subtract the mean

    return dfZeroMean.map(lambda x: np.outer(x, x)).sum() / df.count()


def pca_with_scores(df, k=10):
    """Computes the top `k` principal components, corresponding scores, and all eigenvalues.

    :param df:  A Spark dataframe with a 'features' column, which (column) consists of DenseVectors.
    :param k: The number of principal components to return., defaults to 10
    :return: (eigenvectors, `RDD` of scores, eigenvalues)\n
        * Eigenvectors: multi-dimensional array where the number of\
            rows equals the length of the arrays in the input `RDD` and the number of columns equals`k`.
        * `RDD` of scores: has the same number of rows as `data` and consists of arrays of length `k`.
        * Eigenvalues is an array of length d (the number of features).
    
    :Note:
        All eigenvalues should be returned in sorted order (largest to smallest). `eigh` returns
        each eigenvectors as a column.  This function should also return eigenvectors as columns.
    """
    cov = estimateCovariance(df)
    col = cov.shape[1]
    eigVals, eigVecs = eigh(cov)
    inds = np.argsort(eigVals)
    eigVecs = eigVecs.T[inds[-1:-(col + 1):-1]]
    components = eigVecs[0:k]
    eigVals = eigVals[inds[-1:-(col + 1):-1]]  # sort eigenvals
    score = df.select(df['features']).rdd.map(lambda x: x[0]).map(lambda x: np.dot(x, components.T))
    # Return the `k` principal components, `k` scores, and all eigenvalues

    return components.T, score, eigVals


def varianceExplained(df, k=10):
    """
    Returns the proportion of variance explained by `k` principal componenets. Calls the above PCA procedure
    
    :param df: PySpark DataFrame
    :param k: number of principal components , defaults to 10
    :return: (proportion, principal_components, scores, eigenvalues)
    """
    components, scores, eigenvalues = pca_with_scores(df, k)
    return sum(eigenvalues[0:k]) / sum(eigenvalues), components, scores, eigenvalues


# PCA reconstruction Functions

def reconstructPCA(sql, df, pc, mean, std, originalColumns, fits, pcaColumn='pcaFeatures'):
    """
    Reconstruct data from lower dimensional space after performing PCA
    
    :param sql: SQLContext
    :param df: PySpark DataFrame: inputted PySpark DataFrame
    :param pc: numpy.ndarray: principal components projected onto
    :param mean: numpy.ndarray: mean of original columns
    :param std: numpy.ndarray: standard deviation of original columns
    :param originalColumns: list: original column names
    :param fits: fits of features returned from best_fit_distribution
    :param pcaColumn: column in df that contains PCA features, defaults to 'pcaFeatures'
    :return: dataframe containing reconstructed data
    """

    cols = df.columns
    cols.remove(pcaColumn)

    pddf = df.toPandas()
    first_series = pddf['pcaFeatures'].apply(lambda x: np.array(x.toArray())).as_matrix().reshape(-1, 1)
    first_features = np.apply_along_axis(lambda x: x[0], 1, first_series)
    # undo-ing PCA
    first_reconstructed = np.dot(first_features, pc)
    # undo-ing the scaling
    first_reconstructed = np.multiply(first_reconstructed, std) + mean
    first_reconstructedDF = pd.DataFrame(first_reconstructed, columns=originalColumns)
    for _col in cols:
        first_reconstructedDF[_col] = pddf[_col]

    # This is a pyspark Dataframe containing the reconstructed data, including the dummy columns for the string variables-- next step is to reverse the one-hot-encoding for the string columns
    first_reconstructed = sql.createDataFrame(first_reconstructedDF)

    cols_to_exclude = ['DATE_OF_STUDY']
    postPipeStages, String_Columns = postprocessing_pipeline(df, cols_to_exclude)

    postPipe = Pipeline(stages=postPipeStages)
    out = postPipe.fit(first_reconstructed).transform(first_reconstructed)
    for _col in String_Columns:
        out = out.join(df.select([_col, _col + '_ind']) \
                       .withColumnRenamed(_col + '_ind', _col + '_activeInd'), _col + '_activeInd') \
            .dropDuplicates()
    cols_to_drop = [_col for _col in out.columns if any([base in _col for base in String_Columns]) and '_' in _col]

    reconstructedDF = out.drop(
        *cols_to_drop)  # This is the equivalent as the first translated reconstructed dataframe above
    clip = F.udf(lambda x: x if x > 0 else 0.0, DoubleType())
    for _key in fits.keys():
        if fits[_key]['dist'] == 'EMPIRICAL':
            reconstructedDF = reconstructedDF.withColumn(_key, F.round(clip(F.col(_key))))
        else:
            reconstructedDF = reconstructedDF.withColumn(_key, clip(F.col(_key)))

    return reconstructedDF


class MarkovChain(object):
    def __init__(self, transition_prob):
        """
        Initialize the MarkovChain instance.

        :param: transition_prob: dict\
            A dict object representing the transition\
            probabilities in Markov Chain.\
            Should be of the form:\
                {'state1': {'state1': 0.1, 'state2': 0.4},
                 'state2': {...}}
        """
        self.transition_prob = transition_prob
        self.states = list(transition_prob.keys())  # states that have transitions to the next layer
        # For states in the form <stateN_M> where N is the visit (layer) and M is the cluster in the N-th Layer
        self.max_num_steps = max([int(i.split('state')[1][0]) for i in self.states])

    def get_max_num_steps(self):
        return self.max_num_steps

    def next_state(self, current_state):
        """
        Returns the state of the random variable at the next time
        instance.
        
        :param current_state: The current state of the system.
        :raises: Exception if random choice fails
        :return: next state
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

    def generate_states(self, current_state, no=10, last=True):
        """
        Generates the next states of the system.

        :param current_state: (str) The state of the current random variable.
        :param no: (int) The number of future states to generate.
        :param last: (bool) Do we want to return just the last value
        """
        try:
            if no > self.max_num_steps:
                print('Number of steps exceeds the max number of possible next steps')
                raise Exception('<no> should not exceed {}. The value of <no> was: {}'.format(self.max_num_steps, no))

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

    def rep_states(self, current_state, no=10, num_reps=10):
        """
        Running generate states a bunch of times and returning the final state that happens the most

        :param current_state: (str) The state of the current random variable
        :param no: (int) number of time steps in the future to run
        num_reps int number of times to run the simpca_with_scoresultion forward
        :return state: the most commonly reached state at the end of these runs
        """
        if no > self.max_num_steps:
            print('Number of steps exceeds the max number of possible next steps')
            raise Exception('<no> should not exceed {}. The value of <no> was: {}'.format(self.max_num_steps, no))

        endstates = []
        for _ in range(num_reps):
            endstates.append(self.generate_states(current_state, no=no, last=True))
        return max(set(endstates), key=endstates.count)
