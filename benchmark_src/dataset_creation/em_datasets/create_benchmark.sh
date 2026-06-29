#!/bin/bash

############## adapt these parameters! ###################

# where do you want the raw datasets to be downloaded/saved to?
raw_datasets_dir="./cache/raw_datasets"

# where do you want the created datasets be saved to?
## Foldername needs to be new to not override previously created datasets
output_dir="./cache/row_similarity_data_full"

# limit input tables to a maximum number of rows, leave empty for no limit
table_row_limit=

# TODO: split into tasks, then datasets
datasets=(
"deepmatcher"
"musicbrainz"
"geological-settlements"
)

##########################################################

for dataset in "${datasets[@]}"; do
    python ./benchmark_src/dataset_creation/em_datasets/dataset_creation_src/prepare_em_datasets.py \
        --dataset "$dataset" \
        --output-dir "$output_dir" \
        --raw-datasets-dir "$raw_datasets_dir" \
        ${table_row_limit:+--table-row-limit "$table_row_limit"}
done