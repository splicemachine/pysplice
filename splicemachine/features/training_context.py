class TrainingContext:
    def __init__(self, **kwargs):
        args = {k.lower(): kwargs[k] for k in kwargs} # Make all keys lowercase
        args = {k: args[k].split(',') if 'columns' in k else args[k] for k in args} # Make value a list for specific pkcolumns and contextcolumns because Splice doesn't support Arrays
        self.__dict__.update(args)
