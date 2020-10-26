class TrainingContext:
    def __init__(self, *, pkcolumns, tscolumn, labelcolumn, context_sql, **kwargs):
        self.pk_columns = pkcolumns
        self.ts_column = tscolumn
        self.label_column = labelcolumn
        self.context_sql = context_sql
        args = {k.lower(): kwargs[k] for k in kwargs} # Make all keys lowercase
        args = {k: args[k].split(',') if 'columns' in k else args[k] for k in args} # Make value a list for specific pkcolumns and contextcolumns because Splice doesn't support Arrays
        self.__dict__.update(args)


    def __repr__(self):
        return f'TrainingContext(' \
               f'PKColumns={self.pk_columns}, ' \
               f'TSColumn={self.ts_column}, ' \
               f'LabelColumn={self.label_column}, \n' \
               f'ContextSQL={self.context_sql}'

    def __str__(self):
        return self.__repr__()
