def custom_formatter(value):
    if isinstance(value, str):
        return f"{value[:25] + '...' if len(value) > 25 else value}"
    elif isinstance(value, float):
        return f"{value:.3g}"
    else:
        return value


def process_dataframe(df, row_limit=5, preserve_dtypes=True):
    original_dtypes = df.dtypes.to_dict() if preserve_dtypes else None
    df_limited = df.head(row_limit)
    df_formatted = df_limited.applymap(custom_formatter)
    if preserve_dtypes and original_dtypes:
        for col, dtype in original_dtypes.items():
            if col in df_formatted.columns:
                try:
                    if dtype == 'object':
                        pass
                    else:
                        df_formatted[col] = df_formatted[col].astype(dtype)
                except (ValueError, TypeError):
                    pass

    return df_formatted


def process_dataframes(dataframes_dict, row_limit=5):
    processed_dataframes = {}

    for name, df in dataframes_dict.items():
        processed_dataframes[name] = process_dataframe(df, row_limit)
        print(f"Processed DataFrame: {name}")

    return processed_dataframes