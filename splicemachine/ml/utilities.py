#!/usr/bin/python
# -*- coding: utf-8 -*-
import random
import time
from collections import defaultdict, OrderedDict

import graphviz
import numpy as np
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.sql import Row


class Run(object):
    def __init__(self, *args, **kwargs):
        ERROR = """
        This class has been deprecated and all of
        its components have been integrated into the
        new splicemachine.ml.management.MLManager class.
        """
        raise Exception(ERROR)


def get_confusion_matrix(spark, TP, TN, FP, FN):
    """
    function that shows you a device called a confusion matrix... will be helpful when evaluating.
    It allows you to see how well your model performs
    :param TP: True Positives
    :param TN: True Negatives
    :param FP: False Positives
    :param FN: False Negatives
    """

    row = Row('', 'True', 'False')
    confusion_matrix = spark._wrapped.createDataFrame([row('True', TP, FN),
                                                       row('False', FP, TN)])
    return confusion_matrix


class SpliceBinaryClassificationEvaluator(object):
    """
    A Function that provides an easy way to evaluate models once, or over random iterations
    """

    def __init__(self, spark, label_column='label', prediction_column='prediction',
                 confusion_matrix=True):
        """
        :param label_column: the column in the dataframe containing the correct output
        :param prediction_column: the column in the dataframe containing the prediction
        :param confusion_matrix: whether or not to show a confusion matrix after each input
        """
        self.spark = spark
        self.avg_tp = []
        self.avg_tn = []
        self.avg_fn = []
        self.avg_fp = []
        self.label_column = label_column
        self.prediction_column = prediction_column
        self.confusion_matrix = confusion_matrix

    def input(self, predictions_dataframe):
        """
        Evaluate actual vs Predicted in a dataframe
        :param predictions_dataframe: the dataframe containing the label and the predicition
        """

        pred_v_lab = predictions_dataframe.select(self.label_column,
                                                  self.prediction_column)  # Select the actual and the predicted labels

        self.avg_tp.append(pred_v_lab[(pred_v_lab.label == 1)
                                      & (
                                              pred_v_lab.prediction == 1)].count())  # Add confusion stats
        self.avg_tn.append(pred_v_lab[(pred_v_lab.label == 0)
                                      & (pred_v_lab.prediction == 0)].count())
        self.avg_fp.append(pred_v_lab[(pred_v_lab.label == 1)
                                      & (pred_v_lab.prediction == 0)].count())
        self.avg_fn.append(pred_v_lab[(pred_v_lab.label == 0)
                                      & (pred_v_lab.prediction == 1)].count())

        if self.confusion_matrix:
            get_confusion_matrix(
                self.spark,
                self.avg_tp[-1],
                self.avg_tn[-1],
                self.avg_fp[-1],
                self.avg_fn[-1],
            ).show()

            # show the confusion matrix to the user

    def get_results(self, output_type='dataframe'):
        """
        Return a dictionary containing evaluated results

        :param output_type: either a dataframe or a dict (which to return)
        :return results: computed_metrics (dict) or computed_df (df)
        """

        TP = np.mean(self.avg_tp)
        TN = np.mean(self.avg_tn)
        FP = np.mean(self.avg_fp)
        FN = np.mean(self.avg_fn)

        if self.confusion_matrix:
            get_confusion_matrix(
                self.spark,
                float(TP),
                float(TN),
                float(FP),
                float(FN)
            ).show()

        computed_metrics = OrderedDict({
            'Sensitivity': float(TP) / (TP + FN),
            'Specificity': float(TN) / (FP + TN),
            'Precision': float(TP) / (TP + FP),
            'Negative Predictive Value': float(TN) / (TN + FN),
            'False Positive Rate': float(FP) / (FP + TN),
            'False Discovery Rate': float(FP) / (FP + TP),
            'False Negative Rate': float(FN) / (FN + TP),
            'Accuracy': float(TP + TN) / (TP + FN + FP + TN),
            'F1 Score': float(2 * TP) / (2 * TP + FP + FN),
            'Matthews Correlation Coefficient': float(TP * TN - FP * FN) / np.sqrt(
                (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)),
        })

        if output_type == 'dict':
            return computed_metrics
        else:

            metrics_row = Row(*computed_metrics.keys())
            computed_row = metrics_row(*[float(computed_metrics[i])
                                         for i in computed_metrics.keys()])
            computed_df = self.spark._wrapped.createDataFrame([computed_row])
            return computed_df


class SpliceBaseEvaluator(object):
    """
    Base ModelEvaluator
    """

    def __init__(self, spark, evaluator, supported_metrics, prediction_column="prediction",
                 label_column="label"):
        """
        Constructor for SpliceBaseEvaluator
        :param spark: spark from zeppelin
        :param evaluator: evaluator class from spark
        :param supported_metrics: supported metrics list
        :param prediction_column: prediction column
        :param label_column: label column
        """
        self.spark = spark
        self.ev = evaluator
        self.prediction_col = prediction_column
        self.label = label_column
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
                                                          metric_val=self.avgs[metric][-1]))

    def get_results(self, dict=False):
        """
        Get Results
        :param dict: whether to get results in a dict or not
        :return: dictionary
        """
        computed_avgs = {}
        for key in self.avgs:
            computed_avgs[key] = np.mean(self.avgs[key])

        if dict:
            return computed_avgs

        metrics_row = Row(*self.supported_metrics)
        computed_row = metrics_row(*[float(computed_avgs[i]) for i in self.supported_metrics])
        return self.spark._wrapped.createDataFrame([computed_row])


class SpliceRegressionEvaluator(SpliceBaseEvaluator):
    """
    Splice Regression Evaluator
    """

    def __init__(self, spark, prediction_column="prediction", label_column="label"):
        supported = ['rmse', 'mse', 'r2', 'mae']
        SpliceBaseEvaluator.__init__(self, spark, RegressionEvaluator, supported,
                                     prediction_column=prediction_column, label_column=label_column)


class SpliceMultiClassificationEvaluator(SpliceBaseEvaluator):
    def __init__(self, spark, prediction_column="prediction", label_column="label"):
        supported = ["f1", "weightedPrecision", "weightedRecall", "accuracy"]
        SpliceBaseEvaluator.__init__(self, spark, MulticlassClassificationEvaluator, supported,
                                     prediction_column=prediction_column, label_column=label_column)


def print_horizontal_line(l):
    """
    Print a horizontal line l digits long
    :param l: num
    :return: none
    """
    print(''.join(['-' * l]))


def display(html):
    """
    Display HTML from python in html
    :param html: html string
    :return:
    * note! you may not print out anything in normal python after using this function
    in that cell
    """
    print('%angular')
    print(html)


class DecisionTreeVisualizer(object):
    """
    Visualize a decision tree, either in code like format, or graphviz
    """

    @staticmethod
    def feature_importance(spark, model, dataset, featuresCol="features"):
        """
        Return a dataframe containing the relative importance of each feature
        :param model:
        :param dataframe:
        :param featureCol:
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
            tree_name,
            visual=True,
    ):
        """
        Visualize a decision tree, either in a code like format, or graphviz
        :param model: the fitted decision tree classifier
        :param feature_column_names: column names for features
        :param label_names: labels vector (below avg, above avg)
        :param tree_name: the name you would like to call the tree
        :param visual: bool, true if you want a graphviz pdf containing your file
        :return: none
        """

        tree_to_json = DecisionTreeVisualizer.replacer(model.toDebugString,
                                                       ['feature ' + str(i) for i in
                                                        range(0, len(feature_column_names))],
                                                       feature_column_names)

        tree_to_json = DecisionTreeVisualizer.replacer(tree_to_json,
                                                       ['Predict ' + str(i) + '.0' for i in
                                                        range(0, len(label_names))],
                                                       label_names)
        if not visual:
            return tree_to_json

        dot = graphviz.Digraph(comment='Decision Tree')
        dot.attr(size='7.75,15.25')
        dot.node_attr.update(color='lightblue2', style='filled')
        json_d = DecisionTreeVisualizer.tree_json(tree_to_json)
        dot.format = 'pdf'

        DecisionTreeVisualizer.add_node(dot, '', '', json_d,
                                        realroot=True)
        dot.render('/zeppelin/webapps/webapp/assets/images/'
                   + tree_name)
        print('Successfully uploaded file to Zeppelin Assests on this cluster')
        print('Uploading.')

        time.sleep(3)
        print('Uploading..')
        time.sleep(3)

        print(
                'You can find your visualization at "https://docs.google.com/gview?url=https://<cluster_name>.splicemachine.io/assets/images/' \
                + tree_name + '.pdf&embedded=true#view=fith')

    @staticmethod
    def replacer(string, bad, good):
        """
        Replace every string in "bad" with the corresponding string in "good"
        :param string: string to replace in
        :param bad: array of strings to replace
        :param good: array of strings to replace with
        :return:
        """

        for (b, g) in zip(bad, good):
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
