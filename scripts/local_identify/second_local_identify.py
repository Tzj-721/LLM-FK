
import json
import time
import asyncio
from typing import List, Dict

import yaml
from ptcompletion import OpenAITask
import pandas as pd
from openai import OpenAI, AsyncAzureOpenAI
from openai import AsyncOpenAI
from pymysql import OperationalError, ProgrammingError
from pymysql.cursors import DictCursor
from tqdm import tqdm
import re

from tqdm.asyncio import tqdm_asyncio

import concurrent.futures
import logging

from util.count_util import get_min_max, out_of_value_feature, cover_feature
from util.cut_dataframe import process_dataframes
from util.openai_multi_client import OpenAIMultiOrderedClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("analysis.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
with open('../../config.yaml', 'r') as file:
    config = yaml.safe_load(file)

api_key = config["api_key"]
base_url = config["base_url"]
concurrency = config["concurrency"]
role = config["role"]
model = config["model"]
data_knowledge = config["data_knowledge"]

client = AsyncOpenAI(
    api_key=api_key,
    base_url=base_url
)

def second_local_identify(dataframes, relation_df, column_info_df):
    prompt_df = get_prompt_total(dataframes, relation_df, column_info_df)
    api = OpenAIMultiOrderedClient(
        aclient=client,
        concurrency=concurrency,
        endpoint="chat.completions",
        data_template={
            "model": model,
            "temperature": 0,
        }
    )
    prompts = prompt_df["prompt"].values
    results = []
    def make_requests():
        for prompt in prompts:
            messages = []
            if role:
                messages.append({
                 "role": "system",
                 "content": role})
            messages.append({
                    "role": "user",
                    "content": prompt
                })
            api.request(data={
                "messages": messages
            })

    api.run_request_function(make_requests)
    for result in api:
        if result.failed:
            results.append("Error")
        else:
            result_1 = result.response.choices[0].message.content
            print(result_1)
            results.append(result_1)

    conclusions = []
    confidences = []
    explanations = []

    for result in results:
        if result.startswith("Error:"):
            conclusions.append('Unknown')
            confidences.append(0.0)
            explanations.append(result)
        else:
            conclusion, confidence, explanation = analyse_res(result)
            conclusions.append(conclusion)
            confidences.append(confidence)
            explanations.append(explanation)

    prompt_df['conclusion'] = conclusions
    prompt_df['confidence'] = confidences
    prompt_df['explanation'] = explanations
    prompt_df = prompt_df.drop(columns=["prompt"])
    result_df = prompt_df[prompt_df['conclusion'] == 'Yes']
    result_df.to_csv("./result.csv")
    return result_df

def re_second_local_identify(config, dataframes, relation_df, column_info_df, connections, error_log, true_log):
    prompt_df = get_prompt_total_re(config, dataframes, relation_df, column_info_df, connections, error_log, true_log)
    api_key = config["api_key"]
    base_url = config["base_url"]
    concurrency = config["concurrency"]
    role = config["role"]
    model = config["model"]
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url
    )
    api = OpenAIMultiOrderedClient(
        aclient=client,
        concurrency=concurrency,
        endpoint="chat.completions",
        # concurrency=concurrency,
        data_template={
            "model": model,
            "temperature": 0,
        }
    )
    prompts = prompt_df["prompt"].values
    results = []
    def make_requests():
        for prompt in prompts:
            messages = []
            if role:
                messages.append({
                 "role": "system",
                 "content": role})
            messages.append({
                    "role": "user",
                    "content": prompt
                })
            api.request(data={
                "messages": messages
            })

    api.run_request_function(make_requests)
    for result in api:
        if result.failed:
            results.append("Error")
        else:
            result_1 = result.response.choices[0].message.content
            print(result_1)
            results.append(result_1)

    conclusions = []
    confidences = []
    explanations = []

    for result in results:
        if result.startswith("Error:"):
            conclusions.append('Unknown')
            confidences.append(0.0)
            explanations.append(result)
        else:
            conclusion, confidence, explanation = analyse_res(result)
            conclusions.append(conclusion)
            confidences.append(confidence)
            explanations.append(explanation)

    prompt_df['conclusion'] = conclusions
    prompt_df['confidence'] = confidences
    prompt_df['explanation'] = explanations

    prompt_df = prompt_df.drop(columns=["prompt", "data_type"])
    result_df = prompt_df[prompt_df['conclusion'] == 'Yes']
    result_df.to_csv("./result.csv")
    result_df = process_dataframe(result_df, error_log, true_log)
    return result_df


def process_dataframe(df: pd.DataFrame,
                      error_log: List[Dict],
                      true_log: List[Dict]) -> pd.DataFrame:
    processed_df = df.copy()
    if error_log:
        for error_item in error_log:
            src_parts = error_item["source"].split('.')
            tgt_parts = error_item["target"].split('.')
            mask = (
                    (processed_df['table_A'] == src_parts[0]) &
                    (processed_df['column_A'] == src_parts[1]) &
                    (processed_df['table_B'] == tgt_parts[0]) &
                    (processed_df['column_B'] == tgt_parts[1])
            )
            processed_df = processed_df[~mask]
    if true_log:
        for true_item in true_log:
            src_parts = true_item["source"].split('.')
            tgt_parts = true_item["target"].split('.')

            mask = (
                    (processed_df['table_A'] == src_parts[0]) &
                    (processed_df['column_A'] == src_parts[1]) &
                    (processed_df['table_B'] == tgt_parts[0]) &
                    (processed_df['column_B'] == tgt_parts[1])
            )
            if mask.any():
                processed_df.loc[mask, 'confidence'] = 1.0
                processed_df.loc[mask, 'explanation'] = processed_df.loc[mask, 'explanation'].fillna('') + "--manual"
            else:
                new_row = pd.DataFrame([{
                    'table_A': src_parts[0],
                    'column_A': src_parts[1],
                    'table_B': tgt_parts[0],
                    'column_B': tgt_parts[1],
                    'conclusion': "Yes",
                    'confidence': 1.0,
                    'explanation': "--expert annotation"
                }])
                processed_df = pd.concat([processed_df, new_row], ignore_index=True)

    return processed_df

def analyse_res(result_text):
    conclusion = 'Unknown'
    confidence = 0.0
    explanation = "no explanation"
    try:
        data = json.loads(result_text)
        if isinstance(data, dict):
            return extract_from_json(data)
    except json.JSONDecodeError:
        pass

    json_match = extract_json_from_text(result_text)
    if json_match:
        return extract_from_json(json_match)

    return extract_from_text(result_text)


def extract_json_from_text(text):
    try:
        stack = []
        start_index = -1
        json_objects = []

        for i, char in enumerate(text):
            if char == '{':
                if not stack:
                    start_index = i
                stack.append('{')
            elif char == '}':
                if stack:
                    stack.pop()
                    if not stack and start_index != -1:
                        json_str = text[start_index:i + 1]
                        try:
                            data = json.loads(json_str)
                            if isinstance(data, dict) and 'conclusion' in data:
                                json_objects.append(data)
                        except json.JSONDecodeError:
                            pass
                        start_index = -1
    except Exception:
        pass

    for obj in json_objects:
        if all(key in obj for key in ['conclusion', 'confidence', 'explanation']):
            return obj

    if json_objects:
        return json_objects[0]

    return None


def extract_from_json(data):
    try:
        conclusion = data.get("conclusion", "Unknown")
        confidence = float(data.get("confidence", 0.0))
        explanation = data.get("explanation", "")

        confidence = max(0.0, min(1.0, confidence))

        if conclusion not in ["Yes", "No"]:
            conclusion = "Unknown"

        return conclusion, confidence, explanation
    except Exception as e:
        return 'Unknown', 0.0, ""


def extract_from_text(text):
    conclusion = 'Unknown'
    confidence = 0.0
    explanation = "No explanation"

    try:
        normalized_text = re.sub(r'\s+', ' ', text.strip())
        patterns = [
            r'conclusion\s*[:：]\s*([Yes|No])',
            r'conclusion(?:is|Yes|：)\s*([is|No])',
            r'(?:Conclusion|conclusion)\s*[:：]\s*(Yes|No|YES|NO)',
            r'\b([Yes|No])\b'
        ]

        conf_patterns = [
            r'confidence\s*[:：]\s*([0-9.]+)',
            r'\b([0-9.]+)\s*scores?(?:\s*confidence)?\b'
        ]

        exp_patterns = [
            r'explanation\s*[:：]\s*(.*?)(?=(conclusion|confidence|$))',
            r'analyse\s*[:：]\s*(.*?)(?=(conclusion|confidence|$))',
            r'reason\s*[:：]\s*(.*)'
        ]

        for pattern in patterns:
            match = re.search(pattern, normalized_text)
            if match:
                conclusion = match.group(1).strip()
                if conclusion in ["Yes", "YES"]:
                    conclusion = "Yes"
                elif conclusion in ["No", "NO"]:
                    conclusion = "No"
                break

        for pattern in conf_patterns:
            match = re.search(pattern, normalized_text)
            if match:
                try:
                    confidence = float(match.group(1))
                    confidence = max(0.0, min(1.0, confidence))
                    break
                except ValueError:
                    pass

        for pattern in exp_patterns:
            match = re.search(pattern, normalized_text, re.DOTALL)
            if match:
                explanation = match.group(1).strip()
                explanation = re.sub(r'\s+', ' ', explanation)
                break

    except Exception as e:
        print(f"{str(e)}")

    if conclusion not in ['Yes', 'No']:
        if "Yes" in text and "No" not in text:
            conclusion = "Yes"
        elif "No" in text and "Yes" not in text:
            conclusion = "No"
        else:
            conclusion = 'Unknown'
            confidence = 0.0

    return conclusion, confidence, explanation

def run_llm_task(prompt, config, client, retries=3, retry_delay=2):
    for attempt in range(retries):
        try:
            return run_llm(prompt, config, client)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            raise

def run_llm(question, config, client):
    model_llm = config["model"]
    role = config["role"]

    timeout = config.get("timeout", 30)
    if len(str(role)) == 0:
        completion = client.chat.completions.create(
            model=model_llm,
            messages=[
                # {'role': "system", 'content': role},
                {'role': "user", 'content': question},
            ],
            temperature=0,
            stream=False,
            timeout=timeout
        )
        res = completion.choices[0].message.content

    else:
        completion = client.chat.completions.create(
            model=model_llm,
            messages=[
                {'role': "system", 'content': role},
                {'role': "user", 'content': question},
            ],
            temperature=0,
            stream=False,
            timeout=timeout
        )
        res = completion.choices[0].message.content
        print(res)
    return res

def get_prompt_total(dataframes, relation_df, column_info_df):
    column_info_dict = {}
    for _, row in column_info_df.iterrows():
        key = (row["table"], row["column"])
        value = {
            "database": row["database"],
            "position": row["position"],
            "data_type": row["data_type"],
            "avg_length": row["avg_length"],
            "cardinality": row["cardinality"],
            "is_id/key_suffix": row["is_id/key_suffix"],
            "table_rows": row["table_rows"],
            "is_nullable": row["is_nullable"],
            "is_all_null": row["is_all_null"]
        }
        column_info_dict[key] = value
    relation_df['prompt'] = relation_df.apply(
        lambda row: get_single_prompt(dataframes, row, column_info_dict),
        axis=1
    )
    relation_df.to_csv("relation.csv")
    return relation_df

def get_single_prompt(dataframes, row, column_info_dict):
    table_a = row["Table_A"]
    column_a = row["Column_A"]
    table_b = row["Table_B"]
    column_b = row["Column_B"]
    prompt = f""" You are a database architecture analyst, proficient in designing primary and foreign key 
relationships for relational databases, skilled in inferring data lineage, thoroughly versed in business 
knowledge, and familiar with various entity concepts such as products, music, orders, and other entities. You 
excel at accurately identifying the most likely foreign keys from multiple candidate sets based on naming 
conventions, database modeling experience, and the business entities represented by the database.

Please analyze whether a foreign key dependency exists between two columns from different data tables—i.e., 
whether column A references column B—and return the results in the specified format. Reason step by step based on the 
following information: 

## Database Background 
{data_knowledge}

## Column information
Column A:
{get_one_col_info(table_a, column_a, dataframes, column_info_dict)}

Column B:
{get_one_col_info(table_b, column_b, dataframes, column_info_dict)}

## Decision Perspective
1，Semantics
- Step 1: Understand the real-world entities or relationships represented by the two tables and interpret the meaning of their attributes based on columns.
- Step 2: Based on their semantics, determine whether the foreign key relationship aligns with logical relationships such as "is-a" or "has-a".
- Step 3: Re-evaluate the semantic validity of the foreign key relationship against database design principles, considering the broader database context.

2, Syntax
- Higher string similarity generally indicates a greater likelihood of a foreign key relationship. Analyze the similarity of table/column names for the candidate.
- Columns with typical suffixes (e.g., id, key, no) are more likely to be foreign keys. Assess whether column names conform to such conventions.

3，Data and Distribution Analysis
{get_one_data_analyse(row, dataframes, column_info_dict)}

4，Other Issue Investigation 
- If the unique primary key of one table depends on the unique primary key of another table, such a relationship is generally not considered a foreign key relationship.

## Output
Please comprehensively consider the above perspectives step by step and reach a final conclusion. Output the result in JSON format as follows:
{
"conclusion": ["Yes"/"No"] (whether a foreign key relationship exists),
"confidence": [0.00-1.00] (confidence level based on comprehensive evaluation of the above perspectives),
"explanation": [A 2-3 sentence analysis summary including key evidence, integrating the aforementioned considerations. Keep it in one line without line breaks.]
}

Example:
```json
{{
  "conclusion": "Yes",  
  "confidence": 0.85,  
  "explanation": "Column names include the 'id' suffix, Column B is a primary key, and 98% of Column A values exist in Column B, aligning with the user-order relationship logic."  
}}
```
    """.strip()
    return prompt

def get_prompt_total_re(dataframes, relation_df, column_info_df, error_log, true_log):
    column_info_dict = {}
    error_log_str = explain_error_log(error_log)
    true_log_str = explain_true_log(true_log)
    for _, row in column_info_df.iterrows():
        key = (row["table"], row["column"])
        value = {
            "database": row["database"],
            "position": row["position"],
            "data_type": row["data_type"],
            "avg_length": row["avg_length"],
            "cardinality": row["cardinality"],
            "is_id/key_suffix": row["is_id/key_suffix"],
            "table_rows": row["table_rows"],
            "is_nullable": row["is_nullable"],
            "is_all_null": row["is_all_null"]
        }
        column_info_dict[key] = value
    relation_df['prompt'] = relation_df.apply(
        lambda row: get_single_prompt_re(dataframes, row, column_info_dict, error_log_str, true_log_str),
        axis=1
    )
    relation_df.to_csv("relation.csv")
    return relation_df

def get_single_prompt_re(dataframes, row, column_info_dict, error_log_str, true_log_str):
    table_a = row["table_A"]
    column_a = row["column_A"]
    table_b = row["table_B"]
    column_b = row["column_B"]
    prompt = f"""
You are a database architecture analyst, proficient in designing primary and foreign key 
relationships for relational databases, skilled in inferring data lineage, thoroughly versed in business 
knowledge, and familiar with various entity concepts such as products, music, orders, and other entities. You 
excel at accurately identifying the most likely foreign keys from multiple candidate sets based on naming 
conventions, database modeling experience, and the business entities represented by the database.

Please analyze whether a foreign key dependency exists between two columns from different data tables—i.e., 
whether column A references column B—and return the results in the specified format. Reason step by step based on the 
following information: 

## Database Background 
{data_knowledge}

## Column information
Column A:
{get_one_col_info(table_a, column_a, dataframes, column_info_dict)}

Column B:
{get_one_col_info(table_b, column_b, dataframes, column_info_dict)}

## Decision Perspective
1，Semantics
- Step 1: Understand the real-world entities or relationships represented by the two tables and interpret the meaning of their attributes based on columns.
- Step 2: Based on their semantics, determine whether the foreign key relationship aligns with logical relationships such as "is-a" or "has-a".
- Step 3: Re-evaluate the semantic validity of the foreign key relationship against database design principles, considering the broader database context.

2, Syntax
- Higher string similarity generally indicates a greater likelihood of a foreign key relationship. Analyze the similarity of table/column names for the candidate.
- Columns with typical suffixes (e.g., id, key, no) are more likely to be foreign keys. Assess whether column names conform to such conventions.

3，Data and Distribution Analysis
{get_one_data_analyse(row, dataframes, column_info_dict)}

4，Other Issue Investigation 
- If the unique primary key of one table depends on the unique primary key of another table, such a relationship is generally not considered a foreign key relationship.

## Manually Annotated Foreign Key Relationships (Format: Table.Column)
{true_log_str}{error_log_str}

## Output
Please comprehensively consider the above perspectives step by step and reach a final conclusion. Output the result in JSON format as follows:
{
"conclusion": ["Yes"/"No"] (whether a foreign key relationship exists),
"confidence": [0.00-1.00] (confidence level based on comprehensive evaluation of the above perspectives),
"explanation": [A 2-3 sentence analysis summary including key evidence, integrating the aforementioned considerations. Keep it in one line without line breaks.]
}

Example:
```json
{{
  "conclusion": "Yes",  
  "confidence": 0.85,  
  "explanation": "Column names include the 'id' suffix, Column B is a primary key, and 98% of Column A values exist in Column B, aligning with the user-order relationship logic."  
}}
```
    """.strip()
    return prompt

def get_one_col_info(table_a, column_a, dataframes, column_info_dict):
    dataframes_cut = process_dataframes(dataframes, row_limit=5)
    table_a_df = dataframes_cut[table_a]
    database_a = column_info_dict[(table_a, column_a)]["database"]
    table_block_a = (
        f"\n"
        f"```markdown\n"
        f"{table_a_df.head().to_markdown()}\n" 
        f"```"
    )
    cardinality = column_info_dict[(table_a, column_a)]["cardinality"]
    table_rows = column_info_dict[(table_a, column_a)]["table_rows"]
    min_a, max_a = get_min_max(dataframes, table_a, column_a)
    def calculate_ratio(table_a, column_a):
        cardinality = column_info_dict[(table_a, column_a)]["cardinality"]
        table_rows = column_info_dict[(table_a, column_a)]["table_rows"]
        if table_rows != 0:
            return cardinality / table_rows
        else:
            return 0

    if table_rows == 0:
        prompt_one_info = f"""
- database: {database_a}
- table: {table_a}
- column: {column_a}
- ordinal position of the column in the table: {column_info_dict[(table_a, column_a)]["position"]}
- data type: {column_info_dict[(table_a, column_a)]["data_type"]}
- other information: It is an empty table.
- sample data: {table_block_a}
    """
        return prompt_one_info
    elif cardinality == 0:
        prompt_one_info = f"""
- database: {database_a}
- table: {table_a}
- column: {column_a}
- ordinal position of the column in the table: {column_info_dict[(table_a, column_a)]["position"]}
- data type: {column_info_dict[(table_a, column_a)]["data_type"]}
- table rows：{table_rows}
- Cardinality ratio of distinct values to table row count:0
- other information: The column is empty without data.
- sample data: {table_block_a}
            """
        return prompt_one_info
    else:
        prompt_one_info = f"""
- database: {database_a}
- table: {table_a}
- column: {column_a}
- ordinal position of the column in the table: {column_info_dict[(table_a, column_a)]["position"]}
- data type: {column_info_dict[(table_a, column_a)]["data_type"]}
- average value length: {column_info_dict[(table_a, column_a)]["avg_length"]}
- the number of distinct values: {column_info_dict[(table_a, column_a)]["cardinality"]}
- table rows: {column_info_dict[(table_a, column_a)]["table_rows"]}
- Cardinality ratio of distinct values to table row count: {calculate_ratio(table_a, column_a)}
- min value：{min_a}
- max value：{max_a}
- sample data: {table_block_a}
    """
        return prompt_one_info

def get_one_data_analyse(row, dataframes, column_info_dict):
    res_prompt = ""
    table_a = row["table_A"]
    column_a = row["column_A"]
    table_b = row["table_B"]
    column_b = row["column_B"]
    cardinality_a = column_info_dict[(table_a, column_a)]["cardinality"]
    cardinality_b = column_info_dict[(table_b, column_b)]["cardinality"]
    table_rows_a = column_info_dict[(table_a, column_a)]["table_rows"]
    table_rows_b = column_info_dict[(table_b, column_b)]["table_rows"]
    avg_length_a = column_info_dict[(table_a, column_a)]["avg_length"]
    avg_length_b = column_info_dict[(table_b, column_b)]["avg_length"]

    if table_rows_a == 0 and table_rows_b == 0:
        res_prompt = '''
- Both tables are empty with no data distribution information.
        '''
    elif cardinality_a == 0 and cardinality_b == 0:
        table_div = table_rows_a / table_rows_b
        res_prompt = f'''
- Table size ratio: Generally, a foreign key attribute does not reference a very small subset of the referenced unique key. The ratio of the tuple count in table {table_a} (containing column {column_a}) to the tuple count in table {table_b} (containing column {column_b}) is {table_div}.
- Additional information: Since both columns are empty, no other data distribution information is available.
        '''
    elif table_rows_a == 0:
        res_prompt = f'''
- Table A is empty; its data distribution characteristics cannot be analyzed.
                '''
    elif cardinality_a == 0:
        table_div = table_rows_a / table_rows_b
        res_prompt = f'''
- Table size ratio: Generally, a foreign key attribute does not reference a very small subset of the referenced unique key. The ratio of the tuple count in table {table_a} (containing column {column_a}) to the tuple count in table {table_b} (containing column {column_b}) is {table_div}.
- Additional information: Since Column{column_a} is empty，no other data distribution information is available.
                '''
    else:
        table_div = table_rows_a / table_rows_b
        cover_range = cover_feature(dataframes, row)
        out_of_range = out_of_value_feature(dataframes, row)

        res_prompt = f'''
- Inclusion Dependency: The values in column {column_a} of table {table_a} form a subset of the values in column {column_b} of table {table_b}, indicating a inclusion dependency relationship.
- Coverage Ratio: A foreign key should generally cover a substantial portion of the referenced unique key. The ratio of distinct values in column {column_b} of table {table_b} that fall within the min-max range of column {column_a} in table {table_a} to the total distinct values in column {column_b} is {cover_range}.
- Average Length Difference: The average value lengths of foreign key and referenced key columns are typically similar. The average length of column {column_a} in table {table_a} is {avg_length_a}, while that of column {column_b} in table {table_b} is {avg_length_b}.
- Out-of-Range Ratio: Dependent values should be evenly distributed across the referenced values rather than covering only a narrow range. The ratio of distinct values in column {column_b} of table {table_b} that fall outside the min-max range of column {column_a} in table {table_a} to the total distinct values in column {column_b} is {out_of_range}.
- Table Size Ratio: The ratio of the number of tuples in table {table_a} (containing column {column_a}) to the number of tuples in table {table_b} (containing column {column_b}) is {table_div}.
            '''
    return res_prompt

# extended for ICL
def explain_error_log(error_log):
    if not error_log:
        return ""
    summary = ""
    for i, entry in enumerate(error_log, 1):
        summary += f"False Foreign key #{i}:\n"
        summary += f"  referencing column: {entry['source']}\n"
        summary += f"  referencd column: {entry['target']}\n"
        summary += f"  error explanation: {entry['reason']}\n"
        summary += f"\n"

    prompt = f'''
## False Foreign keys identified by expert
{summary}
    '''
    return prompt

# extended for ICL
def explain_true_log(true_log):
    if not true_log:
        return ""
    summary = ""
    for i, entry in enumerate(true_log, 1):
        summary += f"True Foreign key #{i}:\n"
        summary += f"  referencing column: {entry['source']}\n"
        summary += f"  referencd column: {entry['target']}\n"
        summary += f"  true explanation: {entry['reason']}\n"
        summary += f"\n"

    prompt = f'''
## True Foreign keys identified by expert
{summary}
    '''
    return prompt


