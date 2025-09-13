import os

import pandas as pd

from scripts.global_conflict_resolve.third_confilct_resolve import third_conflict_resolve
from scripts.local_identify.second_local_identify import second_local_identify
from scripts.prune.first_prune_total import first_prune_total
from util.data_util import read_data_raw


def run(data_dir, data_info_path):
    dataframes = read_data_raw(data_dir)
    data_info = pd.read_csv(data_info_path)
    relation_df = first_prune_total(dataframes, data_info)
    result_df = second_local_identify(dataframes, relation_df, data_info)
    fks = third_conflict_resolve(result_df, dataframes)
    fks.to_csv("fks.csv")
    return fks