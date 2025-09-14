import json
import os
import re

import pandas as pd
from itertools import combinations

from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml

from util.cut_dataframe import process_dataframes

with open('../../config.yaml', 'r') as file:
    config = yaml.safe_load(file)

api_key = config["api_key"]
base_url = config["base_url"]
role = config["role"]
model = config["model"]
client = OpenAI(
    api_key=api_key,
    base_url=base_url
)
data_knowledge = config["data_knowledge"]

def unique_key_prune(candidate, dataframes):
    table_b_set = candidate['Table_B'].unique()
    results = {}
    with ThreadPoolExecutor() as executor:
        futures = {}
        for table in table_b_set:
            df = dataframes[table]
            future = executor.submit(process_table, table, df)
            futures[future] = table

        with tqdm(total=len(futures), desc="Processing UCC") as pbar:
            for future in as_completed(futures):
                table_name, ucc_str = future.result()
                results[table_name] = ucc_str
                pbar.update(1)

    result_data = []
    for table, ucc in results.items():
        result_data.append({'table': table, 'ucc': ucc})

    df = pd.DataFrame(result_data)
    df['ucc_prompt'] = df.apply(lambda row: generate_prompt_with_ucc(row, process_dataframes(dataframes)), axis=1)
    df['result'] = df.apply(get_result_ucc, axis=1)
    uk_dict = {}
    for _, row in df.iterrows():
        table_name = row['table']
        result_fields = row['result']
        if ',' in str(result_fields):
            fields = [f.strip() for f in str(result_fields).split(',')]
        else:
            fields = [str(result_fields).strip()]

        uk_dict[table_name] = fields

    filtered_rows = []
    for _, row in candidate.iterrows():
        table_b = row['Table_B']
        column_b = row['Column_B']
        if table_b in uk_dict:
            if column_b in uk_dict[table_b]:
                filtered_rows.append(row)

    filtered_df = pd.DataFrame(filtered_rows)
    return filtered_df

def find_minimal_uccs(df):
    columns = df.columns.tolist()
    minimal_uccs = set()
    current_non_uniques = []

    for col in columns:
        print(df[col].nunique)
        if df[col].nunique() == len(df):
            print(df[col].nunique)
            minimal_uccs.add(frozenset([col]))
        else:
            current_non_uniques.append(frozenset([col]))

    k = 1
    while True:
        next_non_uniques = []
        candidates = []
        one_non_uniques = [frozenset([col]) for col in columns if not df[col].is_unique]
        for base in current_non_uniques:
            for ext in one_non_uniques:
                if not ext.issubset(base):
                    new_candidate = base.union(ext)
                    if len(new_candidate) == k + 1:
                        candidates.append(new_candidate)

        candidates = list(set(frozenset(c) for c in candidates))

        valid_candidates = [
            c for c in candidates
            if not any(ucc.issubset(c) for ucc in minimal_uccs)
        ]

        for candidate in valid_candidates:
            cols = list(candidate)
            if df[cols].duplicated().sum() == 0:
                minimal_uccs.add(frozenset(cols))
            else:
                next_non_uniques.append(frozenset(cols))

        if not next_non_uniques or (k + 1 >= len(columns)):
            break
        current_non_uniques = next_non_uniques
        k += 1

    return [set(ucc) for ucc in minimal_uccs]

def find_all_uccs(dataframes, parallel=False):
    ucc_results = {}
    with tqdm(total=len(dataframes), desc="Processing tables") as pbar:
        if parallel:
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(find_minimal_uccs, df): tbl
                    for tbl, df in dataframes.items()
                }
                for future in futures:
                    result = future.result()
                    tbl = futures[future]
                    ucc_results[tbl] = result if result else None
                    pbar.update(1)
        else:
            for tbl, df in dataframes.items():
                result = find_minimal_uccs(df)
                ucc_results[tbl] = result if result else None
                pbar.update(1)

    return ucc_results


def format_ucc(ucc_set):
    if not ucc_set:
        return "NULL"
    return '; '.join([','.join(sorted(cols)) for cols in ucc_set])


def process_table(table_name, df):
    try:
        ucc = find_minimal_uccs(df)
        return table_name, format_ucc(ucc)
    except Exception as e:
        return table_name, f"Error: {str(e)}"


def generate_prompt_with_ucc(row, dataframes):
    base_prompt = f"You are an expert in the field of data modeling and analysis. Since the referenced item in a " \
                  f"foreign key is a table's unique key, assume that each table can only have one unique key " \
                  f"referenced by other tables. Please help me identify the most likely unique key for each table " \
                  f"that is referenced by other tables. The candidate minimal unique column combinations (UCCs) for " \
                  f"table '{row['table']}' are as follows. The referenced unique key is one of these candidate sets. " \
                  f"If there is only one candidate minimal UCC, then this one should be used.\n"
    principles = f'''
## Database Background
{data_knowledge}
## Selection principles:
1, It can uniquely identify a tuple in the table (since all candidate sets are minimal UCCs, this condition is already satisfied).
2, The ordinal position of the column in the table; generally, the earlier it appears, the more likely it is to be the referenced unique key.
3, The field name contains common naming conventions such as "key" or "id," which align with the conventions for referenced unique keys.
4, The corresponding field's data type is most likely an integer or a string.
5, The length of the corresponding data text is generally short and suitable for human readability.
6, The fewer the number of fields involved in the candidate set, the higher the probability that it is the referenced unique key.
7, The setting of the referenced unique key aligns with business logic based on the entity meaning of the table.

## Candidate solutions:    
    
    '''
    table = row['table']
    df = dataframes[table]
    sample_data = "> sample data：\n```markdown\n" + df.head().to_markdown(index=False) + "\n```\n\n"

    if pd.isna(row['ucc']) or not row['ucc']:
        return base_prompt + "there is no ucc found，please return NULL"

    ucc_list = [s.strip() for s in row['ucc'].split(';')]
    options = '\n'.join([f"{i + 1}. {cols}" for i, cols in enumerate(ucc_list)])

    return f"{base_prompt}{sample_data}{principles}\n{options}\nPlease return only the selected candidate option number (pk) in JSON format. If no referenced unique key is identified, set pk to NULL."


def generate_prompt_without_ucc(row, dataframes):
    base_prompt = f"You are an expert in the field of data modeling and analysis. Since the referenced item in a " \
                  f"foreign key is a table's unique key, assume that each table can only have one unique key " \
                  f"referenced by other tables. Please help me identify the most likely unique key for each table " \
                  f"that is referenced by other tables.\n"
    principles = f'''
## Database Background
{data_knowledge}
## Selection principles:
1, It can uniquely identify a tuple in the table (since all candidate sets are minimal UCCs, this condition is already satisfied).
2, The ordinal position of the column in the table; generally, the earlier it appears, the more likely it is to be the referenced unique key.
3, The field name contains common naming conventions such as "key" or "id," which align with the conventions for referenced unique keys.
4, The corresponding field's data type is most likely an integer or a string.
5, The length of the corresponding data text is generally short and suitable for human readability.
6, The fewer the number of fields involved in the candidate set, the higher the probability that it is the referenced unique key.
7, The setting of the referenced unique key aligns with business logic based on the entity meaning of the table.
   

    '''
    table = row['table']
    df = dataframes[table]
    sample_data = "> sample data：\n```markdown\n" + df.head().to_markdown(index=False) + "\n```\n\n"

    return f"{base_prompt}{sample_data}{principles}\nPlease return only the selected candidate (pk) in JSON format. If no referenced unique key is identified, set pk to NULL."

def run_llm(question):
    if len(str(role)) == 0:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {'role': "user", 'content': question},
            ],
            temperature=0,
            stream=False,
        )
        res = completion.choices[0].message.content

    else:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {'role': "system", 'content': role},
                {'role': "user", 'content': question},
            ],
            temperature=0,
            stream=False,
        )
        res = completion.choices[0].message.content

    return res

def extract_scheme(raw_text, scheme_number):
    scheme_number = extract_number(str(scheme_number))
    schemes_part = raw_text.split("Candidate solutions:")[1].strip()
    scheme_dict = {}
    for line in schemes_part.split("\n"):
        line = line.strip()
        if not line:
            continue
        match = re.match(r'^(\d+)\.?\s*(.*)$', line)
        if match:
            num = match.group(1)
            fields = [f.strip() for f in match.group(2).split(",")]
            scheme_dict[num] = ",".join(fields)
    return scheme_dict.get(str(scheme_number))

def extract_number(text):
    return int(re.sub(r'\D', '', text))

def get_result_ucc(row):
    if pd.isna(row['ucc']) or not str(row['ucc']).strip():
        return ''
    prompt = row['ucc_prompt']
    result = run_llm(prompt)
    print(result)
    try:
        data = json.loads(result)
        pk_value = data.get("pk")
    except json.JSONDecodeError:
        match = re.search(r'\{.*?"pk"\s*:\s*(null|\d+).*?\}', result, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                data = json.loads(match.group())
                pk_value = data.get("pk")
            except (json.JSONDecodeError, AttributeError):
                pk_value = None
        else:
            pk_value = None

    if pk_value is None:
        pk_match = re.search(r'"pk"\s*:\s*(null|\d+)', result, re.IGNORECASE)
        if pk_match:
            pk_str = pk_match.group(1)
            if pk_str.lower() == 'null':
                return ""
            else:
                return pk_str
        else:
            return ""

    if pk_value is None or (isinstance(pk_value, str) and pk_value.lower() == 'null'):
        return ""
    else:
        res = extract_scheme(prompt, pk_value)
        print(res)
        return res

def get_result_no_ucc(row):
    prompt = row['ucc_prompt']
    result = run_llm(prompt)
    print(result)
    if result == "NULL":
        return ""
    else:
        json_start = result.find('{')
        json_end = result.rfind('}') + 1
        json_str = result[json_start:json_end]
        data = json.loads(json_str)
        pk_value = data['pk']
        return pk_value
