import random
from IPython.display import IFrame, HTML, display
from pyspark import SparkContext
from os import environ as env_vars

def hide_toggle(toggle_next=False):
    """
    Function to add a toggle at the bottom of Jupyter Notebook cells to allow the entire cell to be collapsed.
    :param toggle_next: Bool determine if the toggle should affect the current cell or the next cell
    Usage: from splicemachine.stats.utilities import hide_toggle
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
    if run_id and not experiment_id:
        raise Exception('If you are passing in a run id, you must also provide an experiment id!')
    experiment_id = experiment_id or 0
    mlflow_url = '/mlflow/#/experiments/{}'.format(experiment_id)
    if run_id:
        mlflow_url += '/runs/{}'.format(run_id)
    display(HTML('<font size=\"+1\"><a target=\"_blank\" href={}>MLFlow UI</a></font>'.format(mlflow_url)))
    return IFrame(src=mlflow_url, width='100%', height='500px')
  
def get_spark_ui(port=None, spark_session=None):
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
    return IFrame(src=f'/splicejupyter/user/{user}/sparkmonitor/{port}', width='100%', height='500px')
