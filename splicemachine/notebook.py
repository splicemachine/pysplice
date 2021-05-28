import random
from os import environ as env_vars
from IPython import get_ipython
from IPython.display import HTML, IFrame, display
from pyspark import SparkContext
from splicemachine import SpliceMachineException

def _in_splice_compatible_env():
    """
    Determines if a user is using the Splice Machine managed notebooks or not

    :return: Boolean if the user is using the Splice Environment
    """
    try:
        from beakerx import TableDisplay
        import ipywidgets
    except ImportError:
        return False
    return get_ipython()

def run_sql(sql):
    """
    Runs a SQL statement over JDBC from the Splice Machine managed Jupyter notebook environment. If you are running
    outside of the Splice Jupyter environment, you must have a sql kernel and magic set up and configured.

    :param sql: The SQL to execute
    """
    if not get_ipython():
        raise SpliceMachineException("You don't seem to have IPython available. This function is only available"
                                     "in an IPython environment with a configured %%sql magic kernel. Consider using"
                                     "the managed Splice Machine notebook environment")
    get_ipython().run_cell_magic('sql', '', sql)

def hide_toggle(toggle_next=False):
    """
    Function to add a toggle at the bottom of Jupyter Notebook cells to allow the entire cell to be collapsed.
    
    :param toggle_next: Bool determine if the toggle should hide the current cell or the next cell
    
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

    js_f_name = 'code_toggle_{}'.format(str(random.randint(1, 2 ** 64)))

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

def get_mlflow_ui(experiment_id=None, run_id=None):
    """
    Display the MLFlow UI as an IFrame

    :param experiment_id: (int or str) Optionally the experiment id to load into view
    :param run_id: (str) Optionally the run_id to load into view. If you pass in a run_id you must pass an experiment_id
    :return: (IFrame) An IFrame of the MLFlow UI
    """
    if run_id and not experiment_id:
        raise Exception('If you are passing in a run id, you must also provide an experiment id!')
    experiment_id = experiment_id or 0
    mlflow_url = '/mlflow/#/experiments/{}'.format(experiment_id)
    if run_id:
        mlflow_url += '/runs/{}'.format(run_id)
    display(HTML('<font size=\"+1\"><a target=\"_blank\" href={}>MLFlow UI</a></font>'.format(mlflow_url)))
    return IFrame(src=mlflow_url, width='100%', height='700px')

def get_spark_ui(port=None, spark_session=None):
    """
    Display the Spark Jobs UI as an IFrame at a specific port

    :param port: (int or str) The port of the desired spark session
    :param spark_session: (SparkSession) Optionally the Spark Session associated with the desired UI
    :return:
    """
    if port:
        pass
    elif spark_session:
        port = spark_session.sparkContext.uiWebUrl.split(':')[-1]
    elif SparkContext._active_spark_context:
        port = SparkContext._active_spark_context.uiWebUrl.split(':')[-1]
    else:
        raise Exception('No parameters passed and no active Spark Session found.\n'
                        'Either pass in the active Spark Session into the "spark_session" parameter or the port of that session into the "port" parameter.\n'\
                        'You can find the port by running spark.sparkContext.uiWebUrl and taking the number after the \':\'')
    user = env_vars.get('JUPYTERHUB_USER','user')
    display(HTML(f'<font size=\"+1\"><a target=\"_blank\" href=/splicejupyter/user/{user}/sparkmonitor/{port}>Spark UI</a></font>'))
    return IFrame(src=f'/splicejupyter/user/{user}/sparkmonitor/{port}', width='100%', height='700px')
