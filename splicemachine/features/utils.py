def clean_df(df, cols):
    for old,new in zip(df.columns, cols):
        df = df.withColumnRenamed(old,new)
    return df
