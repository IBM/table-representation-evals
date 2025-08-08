#!/bin/bash

############## adapt these parameters! ###################

# where do you want the raw datasets to be downloaded/saved to?
raw_datasets_dir="../../raw_datasets"

# where do you want the created datasets be saved to?
## Foldername needs to be new to not override previously created datasets
output_dir="./created_datasets/row_similarity_data_full"

# limit input tables to a maximum number of rows, set to null if you want to use all rows
table_row_limit=null 

# TODO: split into tasks, then datasets
datasets=(
"deepmatcher"
"musicbrainz"
"geological-settlements"
)

##########################################################

for dataset in "${datasets[@]}"; do
    python dataset_creation_src/create_benchmark.py output_dir=$output_dir dataset=$dataset raw_datasets_dir=$raw_datasets_dir table_row_limit=$table_row_limit
done