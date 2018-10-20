# Splice Machine Python Package
## Installation Instructions: with Pip
`(sudo) pip install git+https://github.com/splicemachine/pysplice --process-dependency-links`

## Installation Instructions: Zeppelin- [x]= format x correctly
```
%spark.pyspark (new zeppelin 0.8) or %pyspark (Zeppelin >0.73)
import os

#install pip without sudo
os.system("wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py --user")

# install splicemachine python package
os.system("""wget https://github.com/splicemachine/pysplice/archive/[pysplice RELEASE_VERSION].zip && 
            unzip [pysplice RELEASE_VERSION].zip && 
            cd pysplice-[pysplice RELEASE VERSION] &&
            ~/.local/bin/pip install . --user""")
```

## Modules
  There are two modules inside the Python splicemachine package currently-- `splicemachine.spark.context` and `splicemachine.ml.zeppelin`
  `splicemachine.spark.context` contains one extremely important class that helps you interact with our database with Spark.
  The `splicemachine.spark.context.PySpliceContext` class is our native spark data source implemented in Python. Currently,
  it offers these, among other functions:
1. df: turn the results of a sql query into dataframe
2. tableExists: returns a boolean of whether or not a table exists
3. getConnection: get a connection to the database (used to renew the connection)
4. insert: insert a dataframe into a table
5. delete: Delete records in a table based on joining by primary keys from the data frame.
6. update: update records in a table based on joining primary keys from the datafrane
7. dropTable: drop a table from database
8. getSchema: return the schema of a table from the database

You can find the source code for this module here ('https://github.com/splicemachine/pysplice/splicemachine/spark/context.py')
```
Usage:
%spark.pyspark
from splicemachine.spark.context import PySpliceContext
splice = PySpliceContext('jdbc:splice://<SOME FRAMEWORK NAME>.splicemachine.io:1527/splicedb;ssl=basic;user=<USER>;password=<PASSWORD>', sqlContext)
my_dataframe = splice.df('SELECT * FROM DEMO.BAKING_CUPCAKES')
filtered_df = my_dataframe.filter(my_dataframe.FLAVOR == 'red_velvet')
# Assume you have created a table with a schema that conforms to filtered_df
splice.insert(filtered_df, 'DEMO.FILTERED_CUPCAKES)
```

  The `splicemachine.ml.zeppelin` package, on the other hand, offers machine learning utilities for use in Splice Machine's Zeppelin notebooks.
  Some of these functions are written specifically for users who are using the MLFlow Splice Machine Lifecycle System, but others are generic for PySpark MLlib.
  
 Here are the functions it offers:

### MLFlow Run Wrapper- cross paragraph logging
 Methods:
 1. Run.create_new_run: remove current run and create a new one
 2. Run.log_param(key: string, value: string): log a parameter to MLFlow with a key: value
 3. Run.log_metric(key:string, metric:numeric): log a metric to MLFlow
 4. Run.log_model(fittedPipeline: FittedPipeline object): log a fitted pipeline for later deployment to SageMaker
 5. Run.log_artifact(run_relative_path: string (path)): log a file to MLFlow (decision tree visualization, model logs etc.)
 
 ```
 Usage:
 from splicemachine.ml.zeppelin import Run
 +---------Cell i-----------+
 baking_run = Run()
 baking_run.create_new_run()
 
 +---------Cell i+1---------+
 baking_run.log_param('dataset', 'banking')
 
 +---------Cell i+2---------+
 baking_run.log_metric('r2', 0.985)
 
 +---------Cell i+3---------+
 fittedPipe = pipeline.fit(baking_df)
 banking_run.log_model(fittedPipe)
 
 +---------Cell i+4---------+
 
 banking_run.log_artifact('output.txt')
 ```
 
### Show Confusion Matrix - Function that shows a nicely formatted confusion matrix for binary classification
1. show_confusion_matrix(sc: spark context from zeppelin, sqlContext: sql context from zeppelin, TP: True positives, TN: true negatves, FP: false positives, FN: False Negatives)
```
Usage:
from splicemachine.ml.zeppelin import show_confusion_matrix
TP = 3
TN = 4
FP = 2
FN = 0
show_confusion_matrix(sc, sqlContext, TP, TN, FP, FN)
---> your confusion matrix will be printed in stdout
```

### Experiment Maker -- Function that creates or uses an MLFlow Experiment
1. experiment_maker(experiment_id: experiment_name (string))

```
Usage:
from splicemachine.ml.zeppelin import experiment_maker
import mlflow

mlflow.set_tracking_uri('/mlruns') # so syncing will work
experiment_name = z.input('Experiment name')
experiment_maker(experiment_name)
```

### Model Evaluator -- Class that Evaluates Binary Classification Models written in PySpark
Methods:
1. __init__(label_column='label': string (input dataframe label col), prediction_column='prediction': string (input dataframe pred col), confusion_matrix=True: bool (show confusion matrix after each input df?))
2. ModelEvaluator.setup_contexts(sc: spark_context from zeppelin, sqlContext: sqlContext from zeppelin) (required to run): Setup Spark Contexts 
3. ModelEvaluator.input(predictions_dataframe: dataframe (containing row with label col and prediction col) # Input a new run to average
4. ModelEvalautor.get_results(output_type: 'dataframe'/'dict') # print out in either dataframe or dict format calculated metrics
Here are the metrics this calculates:
```
computed_metrics = {
   'TPR': float(TP) / (TP + FN),
   'SPC': float(TP) / (TP + FN),
   'PPV': float(TP) / (TP + FP),
   'NPV': float(TN) / (TN + FN),
   'FPR': float(FP) / (FP + TN),
   'FDR': float(FP) / (FP + TP),
   'FNR': float(FN) / (FN + TP),
   'ACC': float(TP + TN) / (TP + FN + FP + TN),
   'F1': float(2 * TP) / (2 * TP + FP + FN),
   'MCC': float(TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)),
 }
 ```
 
 ```
 Usage:
 from splicemachine.ml.zeppelin import ModelEvaluator
 evaluator = ModelEvaluator()
 evaluator.setup_contexts(sc, sqlContext)
 
 CV_ITERATIONS = 5
 
 for _ in range(1, CV_ITERATIONS):
      # do tts, fit pipeline, predict and get df
      evaluator.input(predictions_dataframe)
 
 evaluator.get_results('dataframe').show()
 
```

### Decision Tree Visualizer - Class that allows you to visualize binary/multiclass Decision Tree classification models in PySpark
Methods:
1. DecisionTreeVisualizer.visualize(model: fitted DecisionTreeClassifier model, feature_column_names: list (in order of the features included in your VectorAssembler), label_classes: list (in order of your classes), visual=True (png output via graphviz, or code like structure False))

```
Usage:
from splicemachine.ml.zeppelin import DecisionTreeVisualizer

myfittedDecisionTreeClassifier = unfittedClf.fit(df)
DecisionTreeVisualizer.visualize(myfittedDecisionTreeClassifier, ['flavor', 'color', 'frosting'], ['juicy', 'not juciy'], True)
--> You can see your tree at this URL: <SOME URL>

