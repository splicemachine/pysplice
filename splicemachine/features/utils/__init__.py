def display(obj):
    """
    Helper function to display when in IPython but not have a dependency on IPython

    :param obj: The object to try to display
    :return:
    """
    try:
        from IPython.display import display
        display(obj)
    except:
        print(obj)
