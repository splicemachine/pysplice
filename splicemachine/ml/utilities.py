#!/usr/bin/python
# -*- coding: utf-8 -*-
import random
import time
from collections import defaultdict
from IPython.display import HTML
from collections import defaultdict, OrderedDict

import graphviz
import numpy as np
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegressionModel
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


def experiment_maker(experiment_id):
    ERROR = """
    This function has been deprecated in favor of the splicemachine.ml.management.MLManager class
    """
    raise Exception(ERROR)


class ModelEvaluator():
    def __init__(self, *args, **kwargs):
        ERROR = """
        This class has been renamed to SpliceBinaryClassificationEvaluator
        """
        raise Exception(ERROR)

class SpliceBaseEvaluator(object):
    """
    Base ModelEvaluator
    """

    def __init__(self, spark, evaluator, supported_metrics, predictionCol="prediction",
                 labelCol="label"):
        """
        Constructor for SpliceBaseEvaluator
        :param spark: spark from zeppelin
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


class SpliceBinaryClassificationEvaluator(SpliceBaseEvaluator):
    def __init__(self, spark, predictionCol="prediction", labelCol="label", confusion_matrix = True):
        self.avg_tp = []
        self.avg_tn = []
        self.avg_fn = []
        self.avg_fp = []
        self.confusion_matrix = confusion_matrix

        supported = ["areaUnderROC", "areaUnderPR",'TPR', 'SPC', 'PPV', 'NPV', 'FPR', 'FDR', 'FNR', 'ACC', 'F1', 'MCC']
        SpliceBaseEvaluator.__init__(self, spark, BinaryClassificationEvaluator, supported, predictionCol=predictionCol, labelCol=labelCol)

    def input(self, predictions_dataframe):
        """
        Evaluate actual vs Predicted in a dataframe
        :param predictions_dataframe: the dataframe containing the label and the predicition
        """
        for metric in self.supported_metrics:
            if metric in ['areaUnderROC' , 'areaUnderPR']:
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
            ax.plot(roc['FPR'],roc['TPR'], label = 'Training set areaUnderROC: \n' + str(trainingSummary.areaUnderROC))
            ax.set_ylabel('False Positive Rate')
            ax.set_xlabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend()
            return ax
        else:
            raise NotImplementedError("Only supported for Logistic Regression Models")


class SpliceRegressionEvaluator(SpliceBaseEvaluator):
    """
    Splice Regression Evaluator
    """

    def __init__(self, spark, predictionCol="prediction", labelCol="label"):
        supported = ['rmse', 'mse', 'r2', 'mae']
        SpliceBaseEvaluator.__init__(self, spark, RegressionEvaluator, supported, predictionCol=predictionCol, labelCol=labelCol)


class SpliceMultiClassificationEvaluator(SpliceBaseEvaluator):
    def __init__(self, spark, predictionCol="prediction", labelCol="label"):
        supported = ["f1", "weightedPrecision", "weightedRecall", "accuracy"]
        SpliceBaseEvaluator.__init__(self, spark, MulticlassClassificationEvaluator, supported,
                                     predictionCol=predictionCol, labelCol=labelCol)


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
            visual=False,
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
                                                       [f'Predict: {str(i)}.0' for i in
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
        print('Successfully uploaded file to Zeppelin Assets on this cluster')
        print('Uploading.')

        time.sleep(3)
        print('Uploading..')
        time.sleep(3)

        print(
            'You can find your visualization at "https://docs.google.com/gview?url=https://'
            '<cluster_name>.splicemachine.io/assets/images/' \
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

def inspectTable(spliceMLCtx, sql, topN = 5):
    """Inspect the values of the columns of the table (dataframe) returned from the sql query

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

def hide_toggle(toggle_next=False):
    """
    Function to add a toggle at the bottom of Jupyter Notebook cells to allow the entire cell to be collapsed.
    :param toggle_next: Bool determine if the toggle should affect the current cell or the next cell
    Usage: from splicemachine.ml.utilities import hide_toggle
           hide_toggle()
    """
    this_cell = """$('div.cell.code_cell.rendered.selected')"""
    next_cell = this_cell + '.next()'

    toggle_text = 'Toggle show/hide'  # text shown on toggle link
    target_cell = this_cell  # target cell to control with toggle
    js_hide_current = ''  # bit of JS to permanently hide code in current cell (only when toggling next cell)

    if toggle_next:
        target_cell = next_cell
        toggle_text += ' next cell'
        js_hide_current = this_cell + '.find("div.input").hide();'

    js_f_name = 'code_toggle_{}'.format(str(random.randint(1,2**64)))

    html = """
        <script>
            function {f_name}() {{
                {cell_selector}.find('div.input').toggle();
            }}

            {js_hide_current}
        </script>

        <a href="javascript:{f_name}()"><button style='color:black'>{toggle_text}</button></a>
    """.format(
        f_name=js_f_name,
        cell_selector=target_cell,
        js_hide_current=js_hide_current,
        toggle_text=toggle_text
    )

    return HTML(html)
