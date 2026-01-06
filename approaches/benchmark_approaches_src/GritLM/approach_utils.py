import pandas as pd
from typing import Any, List

def convert_row_to_string(table_row: pd.Series):
    row_string = ""
    for i, (col_name, cell_value) in enumerate(table_row.items()):
        cell_value = str(cell_value)

        row_string += col_name + ": " + cell_value
        if i != len(table_row)-1:
            row_string += "; "

    return row_string

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
