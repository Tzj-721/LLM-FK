import pandas as pd
import yaml

with open('../../config.yaml', 'r') as file:
    config = yaml.safe_load(file)
is_ucc = config["is_ucc"]
is_empty_col = config["is_empty_col"]

def rule_prune(dependencies_df: pd.DataFrame,
                        columns_df: pd.DataFrame) -> pd.DataFrame:
    filtered_df = dependencies_df.copy()
    orig_count = len(filtered_df)

    columns_df["table"] = columns_df["table"].str.lower()
    filtered_df["Table_A"] = filtered_df["Table_A"].str.lower()
    filtered_df["Table_B"] = filtered_df["Table_B"].str.lower()


    valid_tables = set(columns_df["table"].unique())
    col_info_map = columns_df.set_index(["table", "column"]).to_dict(orient="index")
    table_row_map = columns_df.groupby("table")["table_rows"].first().to_dict()

    mask = filtered_df["Table_A"].isin(valid_tables) & filtered_df["Table_B"].isin(valid_tables)
    filtered_df = filtered_df[mask].reset_index(drop=True)


    def is_all_null(table: str, col: str) -> bool:
        return col_info_map.get((table.lower(), col), {}).get("cardinality", 0) == 0
    if not is_empty_col:
        mask = ~filtered_df.apply(
            lambda x: is_all_null(x["Table_A"], x["Column_A"]) | is_all_null(x["Table_B"], x["Column_B"]),
            axis=1
        )
        filtered_df = filtered_df[mask].reset_index(drop=True)
        
    def get_col_type(table: str, col: str) -> str:
        return col_info_map.get((table.lower(), col), {}).get("data_type", "").lower()

    filtered_df["type_a"] = filtered_df.apply(lambda x: get_col_type(x["Table_A"], x["Column_A"]), axis=1)
    filtered_df["type_b"] = filtered_df.apply(lambda x: get_col_type(x["Table_B"], x["Column_B"]), axis=1)
    mask = filtered_df["type_a"] == filtered_df["type_b"]
    filtered_df = filtered_df[mask].reset_index(drop=True)

    invalid_types = {"float", "double", "real", "bool", "boolean", "bit", "decimal"}
    mask = ~filtered_df.apply(
        lambda x: (x["type_a"] in invalid_types) | (x["type_b"] in invalid_types),
        axis=1
    )
    filtered_df = filtered_df[mask].reset_index(drop=True)

    filtered_df = filtered_df.drop(columns=["type_a", "type_b"])
    return filtered_df[["Table_A", "Column_A", "Table_B", "Column_B"]]

