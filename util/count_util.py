import pandas as pd


def get_min_max(raw_data_dict, table, column):
    table_df = raw_data_dict[table].copy()
    table_df[column] = pd.to_numeric(table_df[column], errors='coerce').dropna()
    min_value = table_df[column].min()
    max_value = table_df[column].max()
    return min_value, max_value


def out_of_value_feature(raw_data_dict, row):
    table_A, column_A, table_B, column_B = row[:4]
    table_A_df = raw_data_dict[table_A].copy()
    table_B_df = raw_data_dict[table_B].copy()

    table_A_df[column_A] = pd.to_numeric(table_A_df[column_A], errors='coerce').dropna()
    table_B_df[column_B] = pd.to_numeric(table_B_df[column_B], errors='coerce').dropna()
    if table_A_df[column_A].empty or table_B_df[column_B].empty:
        return 0.0
    min_value = table_A_df[column_A].min()
    max_value = table_A_df[column_A].max()
    out_of_range = table_B_df[(table_B_df[column_B] < min_value) | (table_B_df[column_B] > max_value)]
    out_of_range_count = out_of_range[column_B].nunique()

    count_referenced = table_B_df[column_B].nunique()
    if count_referenced > 0:
        percentage_out_of_range = out_of_range_count / count_referenced
    else:
        percentage_out_of_range = 0 

    return percentage_out_of_range


def cover_feature(raw_data_dict, row):
    table_A, column_A, table_B, column_B = row[:4]
    table_A_df = raw_data_dict[table_A].copy()
    table_B_df = raw_data_dict[table_B].copy()
    table_A_df[column_A] = pd.to_numeric(table_A_df[column_A], errors='coerce')
    table_B_df[column_B] = pd.to_numeric(table_B_df[column_B], errors='coerce')
    if table_A_df[column_A].empty or table_B_df[column_B].empty:
        return 0.0
    dependent_min = table_A_df[column_A].min()
    dependent_max = table_A_df[column_A].max()
    referenced_filtered = table_B_df[(table_B_df[column_B] <= dependent_max) | (table_B_df[column_B] >= dependent_min)]
    number_of_values_in_referenced = referenced_filtered[column_B].nunique()
    count_referenced = table_B_df[column_B].nunique()
    if count_referenced == 0:
        return 0
    return number_of_values_in_referenced / count_referenced
