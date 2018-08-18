import os
import random
import time

import graphviz
import mlflow
import mlflow.spark
import numpy as np
from pyspark.sql import Row

"""
Some utilities for use in Zeppelin when doing machine learning
"""


class Run:
    """
    An abstraction over MLFlow Runs, allowing you to do cross cell runs
    """

    def __init__(self):
        self.run_uuid = None

    @staticmethod
    def handle_handlers(handler, *args, **kwargs):
        if handler == 'param':
            mlflow.log_param(*args, **kwargs)
        elif handler == 'metric':
            mlflow.log_metric(*args, **kwargs)
        elif handler == 'artifact':
            mlflow.log_artifact(*args, **kwargs)
        elif handler == 'spark_model':
            mlflow.spark.log_model(*args, **kwargs)
        else:
            raise Exception(
                "Handler {0} not understood. Please use one in ['param', 'metric', "
                "'artifact', 'spark_model']")

    def log_metadata(self, handler, *args, **kwargs):
        if not self.run_uuid:
            with mlflow.start_run():
                self.run_uuid = (mlflow.active_run().__dict__[
                                 '_info'].__dict__['_run_uuid'])
                print("Logged using handler " + handler)
                Run.handle_handlers(handler, *args, **kwargs)
        else:
            with mlflow.start_run(run_uuid=self.run_uuid):
                Run.handle_handlers(handler, *args, **kwargs)
                print("Logged using handler " + handler)
        return True

    def log_param(self, *args, **kwargs):
        return self.log_metadata('param', *args, **kwargs)

    def log_metric(self, *args, **kwargs):
        return self.log_metadata('metric', *args, **kwargs)

    def log_artifact(self, *args, **kwargs):
        return self.log_metadata('artifact', *args, **kwargs)

    def log_model(self, *args, **kwargs):
        return self.log_metadata('spark_model', *args, **kwargs)

    def create_new_run(self):
        """
        Create a new Run
        :return:
        """
        self.run_uuid = None


def show_confusion_matrix(sc, sqlContext, TP, TN, FP, FN):
    """
    function that shows you a device called a confusion matrix... will be helpful when evaluating. It allows you to see how well your model performs
    :param sc: Spark Context
    :param sqlCtx: SQL Context
    :param TP: True Positives
    :param TN: True Negatives
    :param FP: False Positives
    :param FN: False Negatives
    """
    row = Row('', 'True', 'False')
    confusion_matrix = sqlContext.createDataFrame(
        [row('True', TP, FN), row('False', FP, TN)])
    confusion_matrix.show()


def experiment_maker(experiment_id):
    """
    a function that creates a new experiment if "experiment_name" doesn't exist
    or will use the current one if it already does
    :param experiment_id the experiment name you would like to get or create
    """
    print("Tracking Path " + mlflow.get_tracking_uri())
    found = False
    if not len(experiment_id) in [0, 1]:
        for e in [i for i in mlflow.tracking.list_experiments()]:  # Check all experiments
            if experiment_id == e.name:
                print('Experiment has already been created')
                found = True
                os.environ['MLFLOW_EXPERIMENT_ID'] = str(
                    e._experiment_id)  # use already created experiment

        if not found:
            _id = mlflow.tracking.create_experiment(
                experiment_id)  # create new experiment
            print('Success! Created Experiment')
            os.environ['MLFLOW_EXPERIMENT_ID'] = str(_id)  # use it
    else:
        print("Please fill out this field")


class ModelEvaluator(object):
    """
    A Function that provides an easy way to evaluate models once, or over random iterations
    """

    def __init__(self, label_column='label', prediction_column='prediction', confusion_matrix=True):
        """
        :param sc: Spark Context
        :param sqlContext: SQLContext
        :param label_column: the column in the dataframe containing the correct output
        :param prediction_column: the column in the dataframe containing the prediction
        :param confusion_matrix: whether or not to show a confusion matrix after each input
        """
        self.avg_tp = []
        self.avg_tn = []
        self.avg_fn = []
        self.avg_fp = []
        self.sqlContext = None
        self.sc = None
        self.label_column = label_column
        self.prediction_column = prediction_column
        self.confusion_matrix = confusion_matrix

    def setup_contexts(self, sc, sqlContext):
        """
        Setup contexts for ModelEvaluator
        :param sc: spark context
        :param sqlContext: sql context
        """
        self.sc = sc
        self.sqlContext = sqlContext

    def input(self, predictions_dataframe):
        """
        Evaluate actual vs Predicted in a dataframe
        :param predictions_dataframe: the dataframe containing the label and the predicition
        """

        pred_v_lab = predictions_dataframe.select(self.label_column,
                                                  self.prediction_column)  # Select the actual and the predicted labels

        self.avg_tp.append(pred_v_lab[(pred_v_lab.label == 1) & (
            pred_v_lab.prediction == 1)].count())  # Add confusion stats
        self.avg_tn.append(
            pred_v_lab[(pred_v_lab.label == 0) & (pred_v_lab.prediction == 0)].count())
        self.avg_fn.append(
            pred_v_lab[(pred_v_lab.label == 1) & (pred_v_lab.prediction == 0)].count())
        self.avg_fp.append(
            pred_v_lab[(pred_v_lab.label == 0) & (pred_v_lab.prediction == 1)].count())

        if self.confusion_matrix:
            show_confusion_matrix(self.sc, self.sqlContext, self.avg_tp[-1],
                                  self.avg_tn[-1], self.avg_fp[-1], self.avg_fn[-1])
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
            show_confusion_matrix(self.sc, self.sqlContext, float(TP), float(TN), float(FP), float(FN))
            
        computed_metrics = {
            'TPR': float(TP) / (TP + FN),
            'SPC': float(TN) / (FP + TN),
            'PPV': float(TP) / (TP + FP),
            "NPV": float(TN) / (TN + FN),
            "FPR": float(FP) / (FP + TN),
            "FDR": float(FP) / (FP + TP),
            "FNR": float(FN) / (FN + TP)
            "ACC": float(TP + TN) / (TP + FN + FP + TN),
            "F1":  float(2 * TP) / (2 * TP + FP + FN)
            "MCC": float(float(TP*TN - FP*FN) / np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
        }

        if output_type == 'dict':
            return computed_metrics

        else:
            metrics_row = Row('TPR', 'SPC', 'PPV', 'NPV',
                              'FPR', 'FDR', 'FNR', 'ACC', 'F1', 'MCC')
            computed_row = metrics_row(*[float(i) for i in computed_metrics.values()])
            computed_df = self.sqlContext.createDataFrame([computed_row])
            return computed_df


def print_horizontal_line(l):
    print("".join(['-' * l]))


def display(html):
    print("%angular")
    print(html)


class DecisionTreeVisualizer(object):
    """
    Visualize a decision tree, either in code like format, or graphviz
    """

    @staticmethod
    def visualize(model, feature_column_names, label_names, tree_name, visual=True):
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
                                                        range(0, len(label_names))], label_names)
        if not visual:
            return tree_to_json

        dot = graphviz.Digraph(comment='Decision Tree')
        dot.attr(size="7.75,15.25")
        dot.node_attr.update(color='lightblue2', style='filled')
        json_d = DecisionTreeVisualizer.tree_json(tree_to_json)
        dot.format = 'pdf'

        DecisionTreeVisualizer.add_node(dot, '', '', json_d, realroot=True)
        dot.render('/zeppelin/webapps/webapp/assets/images/' + tree_name)
        print('Successfully uploaded file to Zeppelin Assests on this cluster')
        print('Uploading.')

        time.sleep(3)
        print('Uploading..')
        time.sleep(3)

        print('You can find your visualization at "https://docs.google.com/gview?url=https'
              '://<cluster_name>.splicemachine.io/assets/images/' +
              tree_name + '.pdf&embedded=tru'
              'e#view=fith')

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
    def add_node(dot, parent, node_hash, root, realroot=False):
        """
        Traverse through the .debugString json and generate a graphviz tree
        :param dot: dot file objevt
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
                    DecisionTreeVisualizer.add_node(
                        dot, root['name'], node_id, root['children'][0])
                else:
                    DecisionTreeVisualizer.add_node(
                        dot, root['name'], node_id, root['children'][0])
                    DecisionTreeVisualizer.add_node(
                        dot, root['name'], node_id, root['children'][1])

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
                bl = ' '.join(lines.pop(0).split()[1:]).replace(
                    '(', '').replace(')', '')
                block.append(
                    {'name': bl, 'children': DecisionTreeVisualizer.parse(lines)})

                if lines[0].startswith('Else'):
                    be = ' '.join(lines.pop(0).split()[1:]).replace(
                        '(', '').replace(')', '')
                    block.append(
                        {'name': be, 'children': DecisionTreeVisualizer.parse(lines)})
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
        res = [
            {'name': 'Root', 'children': DecisionTreeVisualizer.parse(data[1:])}]
        return res[0]
