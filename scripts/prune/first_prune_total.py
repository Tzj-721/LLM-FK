import yaml

from scripts.prune.IND_prune import find_single_column_ind
from scripts.prune.rule_prune import rule_prune
from scripts.prune.unique_key_prune import unique_key_prune

with open('../../config.yaml', 'r') as file:
    config = yaml.safe_load(file)
is_ucc = config["is_ucc"]

def first_prune_total(dataframes, dataset_info):
    candidate = find_single_column_ind(dataframes)
    candidate = rule_prune(candidate, dataset_info)
    if is_ucc:
        candidate = unique_key_prune(candidate, dataframes)
    return candidate


