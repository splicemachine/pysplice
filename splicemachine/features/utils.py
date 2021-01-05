def clean_df(df, cols):
    for old,new in zip(df.columns, cols):
        df = df.withColumnRenamed(old,new)
    return df

def dict_to_lower(dict):
    """
    Converts a dictionary to all lowercase keys

    :param dict: The dictionary
    :return: The lowercased dictionary
    """
    return {i.lower():dict[i] for i in dict}
