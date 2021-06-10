from typing import List

class TrainingView:
    def __init__(self, *, pk_columns: List[str], ts_column, label_column, sql_text, name, description,
                 view_id = None, view_version = None, **kwargs):
        self.pk_columns = pk_columns
        self.ts_column = ts_column
        self.label_column = label_column
        self.sql_text = sql_text
        self.name = name
        self.description = description
        self.view_id = view_id
        self.view_version = view_version
        args = {k.lower(): kwargs[k] for k in kwargs} # Make all keys lowercase
        args = {k: args[k].split(',') if 'columns' in k and isinstance(args[k], str) else args[k] for k in args} # Make value a list for specific pkcolumns and contextcolumns because Splice doesn't support Arrays
        self.__dict__.update(args)

    def __repr__(self):
        return f'TrainingView(' \
               f'PKColumns={self.pk_columns}, ' \
               f'TSColumn={self.ts_column}, ' \
               f'LabelColumn={self.label_column}, \n' \
               f'ViewSQL={self.sql_text}, \n' \
               f'ViewID={self.view_id}'

    def __str__(self):
        return self.__repr__()
