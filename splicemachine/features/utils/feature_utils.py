from typing import Dict, Union


def sql_to_datatype(typ: str) -> Dict[str,str]:
    """
    Converts a SQL datatype to a DataType object
    ex:
        sql_to_datatype('VARCHAR(50)') -> dict(data_type= 'VARCHAR',length=50)
        sql_to_datatype('DECIMAL(10,2)') -> dict(data_type= 'DECIMAL',precision=10,scale=2)
    :param typ: the SQL data type
    :return: Dict representing a HTTP friendly datatype
    """
    if isinstance(typ, dict): # Already been converted from server
        return typ
    # If it's a type that has params and those params have been set
    tsplit = typ.split('(')
    if tsplit[0] in ('DECIMAL', 'FLOAT','NUMERIC') and len(tsplit) == 2:
        dtype, params = tsplit
        if ',' in params:
            prec, scale = params.strip(')').split(',')
        else:
            prec, scale = params.strip(')'), None
        data_type = dict(data_type=dtype, precision=prec, scale=scale)
    # If it's a type VARCHAR that has a length
    elif tsplit[0] == 'VARCHAR' and len(tsplit) == 2:
        dtype, length = tsplit
        data_type = dict(data_type=dtype, length=length.strip(')'))
    else:
        data_type = dict(data_type=typ)
    return data_type
