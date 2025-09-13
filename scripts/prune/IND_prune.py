from itertools import product

import pandas as pd


def find_single_column_ind(dataframes):
    column_values = {}
    for tbl, df in dataframes.items():
        column_values[tbl] = {
            col: set(df[col].dropna().unique())
            for col in df.columns
        }

    results = []

    for (tbl_A, df_A), (tbl_B, df_B) in product(dataframes.items(), repeat=2):
        if tbl_A == tbl_B:
            continue

        for col_A in df_A.columns:
            a_vals = column_values[tbl_A][col_A]
            if not a_vals:
                continue
            for col_B in df_B.columns:
                b_vals = column_values[tbl_B][col_B]
                if a_vals.issubset(b_vals):
                    results.append({
                        "Table A": tbl_A,
                        "Column A": col_A,
                        "Table B": tbl_B,
                        "Column B": col_B
                    })

    result_df = pd.DataFrame(results)
    return result_df

