import csv
import json
import re
import unittest
from collections import defaultdict, OrderedDict

import pandas as pd
import yaml
from openai import OpenAI

from util.cut_dataframe import process_dataframes

with open('../../config.yaml', 'r') as file:
    config = yaml.safe_load(file)

api_key = config["api_key"]
base_url = config["base_url"]
concurrency = config["concurrency"]
role = config["role"]
model = config["model"]
data_knowledge = config["data_knowledge"]
is_conflict_del = config["is_conflict_del"]
max_iter = config["max_iter"]
client = OpenAI(
        api_key=api_key,
        base_url=base_url
)


def third_conflict_resolve(fk_df, dataframes):
    if not is_conflict_del:
        columns_to_keep = [
            'table_A', 'column_A', 'table_B', 'column_B', 'explanation'
        ]
        resolved_df = fk_df[columns_to_keep]
        resolved_df.to_csv("final.csv")
        return resolved_df
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    dataframes_cut = process_dataframes(dataframes)
    conflict_issues = find_multi_conflict(fk_df)
    conflict_df = solve_multi_conflict(client, model, role, conflict_issues, dataframes_cut, fk_df)
    resolved_df, removed_edges = resolve_cycle_conflicts_iteratively(conflict_df, dataframes_cut, max_iter)
    columns_to_keep = [
        'table_A', 'column_A', 'table_B', 'column_B', 'explanation'
    ]
    resolved_df = resolved_df[columns_to_keep]
    resolved_df.to_csv("final.csv")
    return resolved_df

def find_multi_conflict(fk_df):
    conflict_dict = defaultdict(set)
    for _, row in fk_df.iterrows():
        a_key = (
            row['table_A'],
            row['column_A']
        )
        b_value = (
            row['table_B'],
            row['column_B']
        )

        conflict_dict[a_key].add(b_value)

    conflict_issues = [(k, v) for k, v in conflict_dict.items() if len(v) > 1]
    return conflict_issues

def solve_multi_conflict(client, model, role, conflict_issues, dataframes, conflict_df):
    indices_to_drop = []
    for conflict_item in conflict_issues:
        (table, column), targets = conflict_item
        prompt = multi_conflict_prompt(dataframes, conflict_item)
        print(prompt)
        llm_response = ask_llm(client, model, role, prompt)
        decision = extract_selected_dependency(llm_response)

        if not decision:
            print(f"Error: {table}.{column}")
            continue

        try:
            parts = decision.split(".")
            if len(parts) != 2:
                raise ValueError(f"Error: {decision}")
            table_r, column_r = parts
            mask = (
                    (conflict_df['table_A'] == table) &
                    (conflict_df['column_A'] == column)
            )

            conflict_rows = conflict_df[mask]
            keep_mask = (
                    (conflict_rows['table_B'] == table_r) &
                    (conflict_rows['column_B'] == column_r)
            )
            if not keep_mask.any():
                continue
            drop_indices = conflict_rows[~keep_mask].index.tolist()
            indices_to_drop.extend(drop_indices)

        except Exception as e:
            print(f"Error: {str(e)}")

    if indices_to_drop:
        conflict_df = conflict_df.drop(indices_to_drop).reset_index(drop=True)

    return conflict_df


def ask_llm(client, model, role, prompt):
    messages = []
    if role:
        messages.append({'role': "system", 'content': role})
    messages.append({
                'role': "user",
                'content': prompt,
            })
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        stream=False,
    )
    res = response.choices[0].message.content
    print(res)
    return res

def multi_conflict_prompt(dataframes, conflict_item):
    try:
        (table, column), targets = conflict_item
    except (ValueError, TypeError) as e:
        raise ValueError(f"Error: {conflict_item}") from e

    dep_table_content = (
        f"The sample data of referencing table `{table}`(related column`{column}`)：\n"
        f"```markdown\n{dataframes[table].head().to_markdown()}\n```"
    )

    target_contents = []
    for tgt_table, tgt_column in targets:
        df_sample = dataframes[tgt_table].head()
        table_block = (
            f"\nThe sample data of referenced table `{tgt_table}` (related column `{tgt_column}`）:\n" 
            f"```markdown\n"
            f"{df_sample.to_markdown()}\n"
            f"```"
        )
        target_contents.append(table_block)
    options_list = '\n'.join([f'- {t}.{c}' for t, c in targets])
    target_content_str = "\n\n".join(target_contents)
    prompt = f"""
You are a database architecture analyst with expertise in designing foreign key relationships for relational databases. You are skilled in data lineage inference, possess comprehensive business knowledge, and are familiar with various entity concepts such as products, music, orders, and other entities. You excel at accurately identifying the most likely foreign keys from multiple candidate sets based on naming conventions, database modeling experience, and the business entities represented by the database.
Your current task is to resolve database column reference conflicts. A foreign key in a table can only reference one unique key from another table and cannot reference multiple keys simultaneously. Please select the correct  foreign key to retain based on the following information.

## Conflict Description：
The column {column} in table {table} is simultaneously referenced by multiple external columns:
Referenced columns (format: TableName.ColumnName): 
{options_list}

## Related Table Content：
{dep_table_content}
{target_content_str}

## Chain of Thought
Step 1: Assign a score to each potential foreign key relationship based on semantic relevance strength and database design best practices.
Step 2: Sort the relationships by their assigned scores in descending order.
Step 3: Retain the relationship with the highest score.
Step 4: Re-validate that the resulting references structure remains semantically optimal after processing.

## Output:
Return the selection result in JSON format:{{"selected": "target_table.target_column"}}
Example output:
{{
    "selected": "login_log.id"
}}
    """.strip()

    return prompt


def extract_selected_dependency(response_text):
    try:
        data = json.loads(response_text)
        return data.get("selected")
    except json.JSONDecodeError:
        pass
    try:
        json_match = re.search(r'\{[\s\S]*?\}', response_text)
        if json_match:
            data = json.loads(json_match.group(0))
            return data.get("selected")
    except (json.JSONDecodeError, TypeError):
        pass
    selected_patterns = [
        r'"selected"\s*:\s*"([^"]+)"',
        r'selected\s*:\s*"([^"]+)"',
        r'selected\s*=\s*"([^"]+)"',
    ]

    for pattern in selected_patterns:
        match = re.search(pattern, response_text)
        if match:
            return match.group(1)

    dependency_pattern = r'([\w\-]+)\.([\w\-]+)'
    match = re.search(dependency_pattern, response_text)
    if match:
        return f"{match.group(1)}.{match.group(2)}"

    example_pattern = r'login_log\.id'
    match = re.search(example_pattern, response_text)
    if match:
        return match.group(0)

    return None


def detect_cycles_conflicts(df):
    graph = defaultdict(set)
    edge_details = {}
    all_nodes = set()

    for _, row in df.iterrows():
        src_node = (
            row['table_A'],
            row['column_A']
        )

        tgt_node = (
            row['table_B'],
            row['column_B']
        )

        graph[src_node].add(tgt_node)
        all_nodes.add(src_node)
        all_nodes.add(tgt_node)

        edge_key = (src_node, tgt_node)
        edge_details[edge_key] = {
            'explanation': row.get('explanation', '')
        }

    visited = defaultdict(int)
    path = []
    cycle_found = None

    def dfs(node):
        nonlocal cycle_found

        if visited[node] == 1:
            cycle_start = path.index(node)
            cycle_nodes = path[cycle_start:] + [node]
            min_node = min(cycle_nodes)
            min_index = cycle_nodes.index(min_node)
            normalized_cycle = cycle_nodes[min_index:-1] + cycle_nodes[:min_index + 1]

            cycle_edges = []
            for i in range(len(normalized_cycle) - 1):
                src = normalized_cycle[i]
                tgt = normalized_cycle[i + 1]
                edge_key = (src, tgt)

                if edge_key in edge_details:
                    edge_info = edge_details[edge_key]
                    cycle_edges.append({
                        'source': f"{src[0]}.{src[1]}",
                        'target': f"{tgt[0]}.{tgt[1]}",
                        'explanation': edge_info['explanation']
                    })

            if cycle_edges:
                cycle_found = {
                    'nodes': [f"{node[0]}.{node[1]}" for node in normalized_cycle],
                    'edges': cycle_edges,
                    'full_path': normalized_cycle
                }
                return True
            return False

        if visited[node] == 2:
            return False

        visited[node] = 1
        path.append(node)

        for neighbor in graph.get(node, []):
            if dfs(neighbor):
                return True

        path.pop()
        visited[node] = 2
        return False

    for node in all_nodes:
        if visited[node] == 0:
            if dfs(node):
                return cycle_found

    return False


def format_cycle_for_output(cycle):
    if not cycle:
        return "no cycle"

    nodes = " → ".join(cycle['nodes'])

    edge_details = []
    for i, edge in enumerate(cycle['edges'], 1):
        edge_info = f"{edge['source']} → {edge['target']}"
        # if edge['explanation']:
        #     edge_info += f" - explanation: {edge['explanation']}"
        edge_details.append(f"{i}. {edge_info}")

    return f"detect cycles: {nodes}\n" + "\n".join(edge_details)


def resolve_cycle_conflicts_iteratively(df, dataframes, max_iter=100):

    resolved_df = df.copy()
    removed_edges = []

    for iteration in range(max_iter):
        cycle = detect_cycles_conflicts(resolved_df)

        if not cycle:
            return resolved_df, removed_edges

        prompt = generate_cycle_prompt(cycle, dataframes)
        print(prompt)
        llm_response = ask_llm(client, model, role, prompt)

        try:
            deleted_edge = extract_deleted_edge(llm_response)
        except (json.JSONDecodeError, KeyError) as e:
            continue

        try:
            src_part, tgt_part = deleted_edge.split(' → ')
            src_table, src_col = src_part.split('.')
            tgt_table, tgt_col = tgt_part.split('.')
        except Exception as e:
            continue

        mask = (
                (resolved_df['table_A'] == src_table) &
                (resolved_df['column_A'] == src_col) &
                (resolved_df['table_B'] == tgt_table) &
                (resolved_df['column_B'] == tgt_col)
        )

        if mask.any():
            removed_row = resolved_df[mask].iloc[0]
            removed_edge_info = {
                'src': f"{src_table}.{src_col}",
                'tgt': f"{tgt_table}.{tgt_col}",
            }
            removed_edges.append(removed_edge_info)
            resolved_df = resolved_df[~mask].reset_index(drop=True)

    if detect_cycles_conflicts(resolved_df):
        print(f"{max_iter} over")
    else:
        print(resolved_df)
        return resolved_df, removed_edges

    print(resolved_df)
    return resolved_df, removed_edges


def generate_cycle_prompt(cycle, dataframes):
    cycle_path = " → ".join([
        f"{node[0]}.{node[1]}"
        for node in cycle['full_path']
    ])

    candidate_edges = []
    for i, edge in enumerate(cycle['edges'], 1):
        src_node = edge['source']
        tgt_node = edge['target']
        s_table, s_col = src_node.split(".")
        t_table, t_col = tgt_node.split(".")
        src_info = f"{s_table}.{s_col}"
        tgt_info = f"{t_table}.{t_col}"
        edge_info = f"{src_info} → {tgt_info}"
        candidate_edges.append(f"{i}. {edge_info}")

    table_contents = []
    unique_tables = OrderedDict()

    for node in cycle['full_path']:
        table_identifier = f"{node[0]}"
        unique_tables[table_identifier] = node

    for table_identifier, node in unique_tables.items():
        table, column = node
        df = dataframes[f"{table}"]
        table_block = (
                    f"### Individual Related Table and Column Information\n"
                    f"- table: `{table}`\n"
                    f"- column: `{column}`\n"
                    f"## sample data\n"
                    f"```markdown\n{df.head().to_markdown(index=False)}\n```\n"
            )
        table_contents.append(table_block)
    prompt = f"""
## Task Description
You are a database architecture analyst with expertise in designing foreign key relationships for relational databases. You are skilled in data lineage inference, possess comprehensive business knowledge, and are familiar with various entity concepts such as products, music, orders, and other entities. You excel at accurately identifying the most likely foreign keys from multiple candidate sets based on naming conventions, database modeling experience, and the business entities represented by the database.
Your current task is to resolve circular references conflicts in database columns. Foreign key reference relationships between tables must not form closed loops. Please select one candidate foreign key relationship to remove in order to break the circular chain based on the following information.

## Circular References Description
{cycle_path}

## Candidate References
Please evaluate the following reference relationships and select the one that is least likely to represent an actual foreign key relationship for removal:
""" + "\n".join(candidate_edges) + """

## Related Table Contents
""" + "\n\n".join(table_contents) + """

## Chain of Thought
Step 1: Assign a score to each potential foreign key relationship based on semantic relevance strength and database design best practices.
Step 2: Sort the relationships by their assigned scores in descending order.
Step 3: Remove the relationship with the lowest score.
Step 4: Re-validate that the resulting references structure remains semantically optimal after processing.

## Output
Return the  deleted edge in JSON format:{{"deleted_edge": "target_table.target_column"}}
Example output:
{{
    "deleted_edge": "users.department_id → department.id"
}}

    """.strip()

    return prompt


def extract_deleted_edge(response_text):
    try:
        data = json.loads(response_text)
        return data.get("deleted_edge")
    except json.JSONDecodeError:
        pass

    try:
        json_match = re.search(r'\{[\s\S]*?\}', response_text)
        if json_match:
            data = json.loads(json_match.group(0))
            return data.get("deleted_edge")
    except (json.JSONDecodeError, TypeError):
        pass

    # 尝试3: 使用正则表达式直接查找deleted_edge值
    deleted_edge_patterns = [
        r'"deleted_edge"\s*:\s*"([^"]+)"',
        r'deleted_edge\s*:\s*"([^"]+)"',
        r'deleted_edge\s*=\s*"([^"]+)"',
    ]

    for pattern in deleted_edge_patterns:
        match = re.search(pattern, response_text)
        if match:
            return match.group(1)

    full_dependency_pattern = r'([\w\-]+)\.([\w\-]+)\s*→\s*([\w\-]+)\.([\w\-]+)'
    match = re.search(full_dependency_pattern, response_text)
    if match:
        src_part = f"{match.group(1)}.{match.group(2)}"
        tgt_part = f"{match.group(3)}.{match.group(4)}"
        return f"{src_part} → {tgt_part}"


    simple_dependency_pattern = r'([\w\-]+)\.([\w\-]+)\s*→\s*([\w\-]+)\.([\w\-]+)'
    match = re.search(simple_dependency_pattern, response_text)
    if match:
        return f"{match.group(1)}.{match.group(2)} → {match.group(3)}.{match.group(4)}"

    example_pattern = r'users\.department_id → departments\.id'
    match = re.search(example_pattern, response_text)
    if match:
        return match.group(0)

    return None
