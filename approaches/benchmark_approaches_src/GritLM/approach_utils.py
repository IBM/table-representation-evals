import pandas as pd

def convert_row_to_string(table_row: pd.Series):
    row_string = ""
    for i, (col_name, cell_value) in enumerate(table_row.items()):
        cell_value = str(cell_value)

        row_string += col_name + ": " + cell_value
        if i != len(table_row)-1:
            row_string += "; "

    return row_string