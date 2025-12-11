from typing import List, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from benchmark_src.approach_interfaces.base_interface import BaseTabularEmbeddingApproach
from benchmark_src.approach_interfaces.column_embedding_interface import ColumnEmbeddingInterface
from benchmark_src.approach_interfaces.row_embedding_interface import RowEmbeddingInterface


def convert_row_to_string(table_row: pd.Series):
    row_string = ""
    for i, (col_name, cell_value) in enumerate(table_row.items()):
        cell_value = str(cell_value)

        row_string += col_name + ": " + cell_value
        if i != len(table_row)-1:
            row_string += "; "

    return row_string


def get_unique_values(column):
    #return list(filter(lambda v: type(v) is str, distinct_values))
    return list([str(x) for x in column.unique()])


def split_into_cell_values(embedding_model, token_ids, tokenizer, str_sep):
    #print("--- split_into_cell_values called")
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    text = tokenizer.convert_tokens_to_string(tokens)
    #print('text', text)
    #print(list(zip(range(len(tokens)), token_ids, tokens)))
    special_tokens = [str_sep, tokenizer.bos_token, tokenizer.eos_token, tokenizer.sep_token, tokenizer.cls_token, tokenizer.pad_token]

    embedding_vectors = embedding_model.encode([text], output_value='token_embeddings', show_progress_bar=False)[0].cpu().detach().numpy()

    # collect indices of tokens between seperators (each index range starts and ends with the sep token)
    indices = []
    start = 0
    for i in range(0, len(tokens)):
        if tokens[i] == str_sep:
            indices.append((start, i))
            start = i

    #print('cell split_ indices', indices)
    # assumption string ends with separator
    res = []
    cell_vals = []
    for x in indices:
        start = x[0]
        end = x[1]
        filtered_tokens = [tokens[x] for x in range(start, end) if tokens[x] not in special_tokens]
        #print('filtered tokens', filtered_tokens)
        text = tokenizer.convert_tokens_to_string(filtered_tokens).strip()

        if text == '':
            continue
        cell_vals.append(text)

        # create cell value embedding by taking the mean of the token embeddings
        res.append(np.mean(embedding_vectors[start:end], axis=0))

    # putting this assertion earlier for tokens and embedding vectors causes things to fail
    assert len(cell_vals) == len(res), str(cell_vals) + '-' + str(res)

    #print('tokens for cell values:', cell_vals)
    return res, cell_vals


def get_row_embedding_from_cols(table_row: pd.Series, embedding_lookup: dict):
    all_cell_embeddings = []
    for col_name, cell_value in table_row.items():
        cell_value = str(cell_value)
        try:
            embedding = embedding_lookup[col_name][cell_value]
            all_cell_embeddings.append(embedding)
        except KeyError:
            print('cannot find', col_name, cell_value)

    row_embedding = np.mean(all_cell_embeddings, axis=0)
    return row_embedding

def get_column_values(column, column_name):
    distinct_string_values = get_unique_values(column)
    distinct_string_values = [x for x in distinct_string_values if len(x) > 0]
    return column_name + ":" + ' | '.join(distinct_string_values)

def process_column(embedding_model, column: pd.Series, column_name: str):
    sep = '|' # use of any other separator makes it hard to keep tokenized values and produced token embeddings in synch!
    replacement = '/' # replace sep in the data with the replacement

    distinct_string_values = get_unique_values(column)
    empty_strings = [x for x in distinct_string_values if len(x.strip()) == 0]
    assert len(empty_strings) == 0, f"Please remove strings that are only blank from the data, found {empty_strings}"
    original_string_values = distinct_string_values
    distinct_string_values = [s.replace(sep, replacement) for s in distinct_string_values if s]

    distinct_string_values = [column_name + ":" + s for s in distinct_string_values]

    tokenizer = embedding_model.tokenizer
    sep_str = ' ' + sep + ' '
    # separator token gets broken out into 3 tokens CLS token SEP - so get the index of the middle token
    sep_token_ids = tokenizer(sep_str, return_tensors='pt')['input_ids'][0].cpu().detach().numpy()
    sep_tokens = tokenizer.convert_ids_to_tokens(sep_token_ids)
    #print('sep_tokens', sep_tokens)
    #print('sep token ids', sep_token_ids)

    #### Each model will tokenize everything differently!
    sep_index = -1
    for idx, x in enumerate(sep_tokens):
        if sep in x:
            sep_index = idx
            break
    assert sep_index != -1
    sep_token_id = sep_token_ids[sep_index]
    #print('sep token id', sep_token_id)

    # concatenate all column values to a string
    text = sep_str.join(distinct_string_values) + ' ' + sep
    #print('TEXT', text)
    # tokenize the whole text string
    arr = tokenizer(text, return_tensors='pt')['input_ids'][0].cpu().detach().numpy() # assumption is always batch size of 0

    max_length = embedding_model.max_seq_length # this is actually max position embeddings for the model, not dimensionality of the embeddings
    #print("max length is", max_length)

    res = []
    cell_length_exceeded = False
    if len(arr) < max_length:
        res.append(arr)
    else:
        #logger.debug(f'need splitting cell values because arr is greater than max_length {max_length}')
        indices = [0]
        for i in range(len(arr)):
            if arr[i] == sep_token_id:
                indices.append(i)
        # ensure we get at least 3 values for a column to get sufficient context, if a single value
        # exceeds a token, then truncate each value so we get tokens
        mod_indices = []
        max_cell_len = int(max_length / 3) - 2 # transformers add CLS and end token sometimes
        #print('max_cell_len', max_cell_len)
        for idx in range(1, len(indices)):
            x = indices[idx]
            if x - indices[idx - 1] >= max_cell_len:
                mod_indices.append((indices[idx - 1], indices[idx - 1] + max_cell_len))
                cell_length_exceeded = True
            else:
                mod_indices.append((indices[idx - 1], x))

        # write out chunks now to res, maintaining which col this was derived from
        #print('indices', indices)
        #print('mod_indices', mod_indices)
        vals_so_far = []
        for ix, x in enumerate(mod_indices):
            start = x[0]
            end = x[1]
            length_so_far = len(vals_so_far)
            if ((end - start) + length_so_far) < max_length:
                vals_so_far.extend(arr[start:end])
            else:
                vals_so_far.append(sep_token_id)
                res.append(vals_so_far)
                vals_so_far = []
                vals_so_far.extend(arr[start:end])

        # add remaining lines
        vals_so_far.append(sep_token_id)
        res.append(vals_so_far)

    cell_value_embeddings = []
    all_cell_tokens = []

    for line in res:
        cell_vectors, cell_tokens = split_into_cell_values(embedding_model, line, tokenizer, sep_tokens[sep_index])
        all_cell_tokens.extend(cell_tokens)
        cell_value_embeddings.extend(cell_vectors)

    assert len(cell_value_embeddings) == len(all_cell_tokens)

    #print('cell value embeddings length', len(cell_value_embeddings))
    #print('len distinct string values', len(distinct_string_values))

    contains_out_of_disbn_chars = False
    if not cell_length_exceeded:
        subt = [(idx, item) for idx, item in enumerate(distinct_string_values) if item.lower().strip() not in all_cell_tokens]
        if len(subt) != 0:
            # this often happens because after decoding the tokenized string will still not be exactly the same as before encoding
            print('tokenizer error')
    else:
        subt = []
        for idx, item in enumerate(distinct_string_values):
            if not item.strip().startswith(all_cell_tokens[idx][0:max_cell_len-10]):
                #for idy, x in enumerate(all_cell_tokens[idx]):
                #    if item.strip()[idy] != x:
                #        print('NO MATCH AT', idy, x, item.strip()[idy])
                subt.append((idx, item))

    assert len(cell_value_embeddings) == len(distinct_string_values), str(len(cell_value_embeddings)) + ' ' + str(len(distinct_string_values)) + ' ' + str(all_cell_tokens) + ' ' + str(distinct_string_values)

    lookup_dict = {}

    assert len(original_string_values) == len(set(original_string_values))
    assert len(distinct_string_values) == len(set(distinct_string_values))


    if not len(original_string_values) == len(distinct_string_values):
        print(f"Len orig: {len(original_string_values)} but len distinct: {len(distinct_string_values)}")
        print(set(original_string_values).difference(set(distinct_string_values)))
        raise NotImplementedError

    for i, value in enumerate(original_string_values):
        lookup_dict[value] = cell_value_embeddings[i].tolist()

    return lookup_dict

def create_preprocessed_data(input_table: pd.DataFrame, component:BaseTabularEmbeddingApproach):
    if isinstance(component, RowEmbeddingInterface):
        """
        Linearize the rows into strings
        """
        # convert all rows to strings

        all_rows = []
        for _, row in tqdm(input_table.iterrows()):
            table_row_string = convert_row_to_string(row)
            all_rows.append(table_row_string)
        preprocessed_data = all_rows  # return the preprocessed_data in which ever format you like

    elif isinstance(component, ColumnEmbeddingInterface):
        all_columns = {}
        for c in tqdm(input_table.columns):
            all_columns[c] = get_column_values(input_table[c], c)
        preprocessed_data = all_columns

    return preprocessed_data

def convert_df_to_markdown(df: pd.DataFrame) -> str:
    # Build a list of lines, then join at the end.
    lines: List[str] = []

    headers = [str(header) for header in df.columns]
    lines.append("| " + " | ".join(headers) + " |")

    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in df.values:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")

    return "\n".join(lines) + "\n"


def convert_array_to_markdown(table_array: List[List[Any]], max_rows: int) -> str:
    """
    Converts a list of lists (where the first list is headers)
    into a Markdown table string.

    Args:
        table_array: A list of lists.
                     Example: [["col1", "col2"], ["data1", "data2"]]
        max_rows: The maximum number of data rows to include. If -1, there is no limit.
    """
    if not table_array or not table_array[0]:
        print("ERROR: Empty table array provided.") #TODO: change the whole file to use logging
        return ""

    headers = [str(h) for h in table_array[0]]
    lines: List[str] = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]

    data_rows = table_array[1:]

    if max_rows != -1 and len(data_rows) > max_rows:
        data_rows = data_rows[:max_rows]
        # print(f"Limiting table (with {len(table_array)-1} rows) to first {max_rows} rows.")

    for row in data_rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")

    # Join all lines with a newline and add a final newline
    return "\n".join(lines) + "\n"

