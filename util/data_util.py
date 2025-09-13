import os
import pandas as pd


def read_data_raw(raw_data):
    raw_data_dict = {}
    for filename in os.listdir(raw_data):
        if filename.endswith(".csv"):
            file_path = os.path.join(raw_data, filename)
            try:
                df = pd.read_csv(file_path)
                if df.empty:
                    file_key = os.path.splitext(filename)[0]
                    raw_data_dict[file_key] = df
                    continue
                file_key = os.path.splitext(filename)[0]
                raw_data_dict[file_key] = df
            except pd.errors.EmptyDataError:
                print(f"Skipping file {filename} because it is empty or malformed.")
                continue

    if not raw_data_dict:
        raise ValueError("no csv")

    return raw_data_dict