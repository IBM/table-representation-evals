#!/bin/bash
set -e

pushd .

cd ContextAwareJoin/datasets/

#bash download.sh # (downloads all datasets from ContextAwareJoin)

# Open Data
if [ -d "./opendata" ] && [ "$(ls -A ./opendata)" ]; then
    echo "the opendata dataset was already downloaded."
else
    echo "./opendata is empty or does not exist."
    echo "Downloading OpenData"
    curl -L -O https://zenodo.org/records/15881731/files/opendata-contextawarejoins.zip
    unzip opendata-contextawarejoins.zip
    rm -rf ./opendata
    mv opendata-contextawarejoins opendata
    rm opendata-contextawarejoins.zip
    echo "OpenData Setup Done"
fi

# Valentine
if [ -d "./valentine" ] && [ "$(ls -A ./valentine)" ]; then
    echo "the valentine dataset was already downloaded"
else
    echo "./valentine is empty or does not exist."
    curl -L -o Valentine-datasets.zip "https://zenodo.org/records/5084605/files/Valentine-datasets.zip?download=1"
    unzip Valentine-datasets.zip
    rm -rf ./valentine
    mv Valentine-datasets valentine
    for f in ./valentine/*/Semantically-Joinable/*/*mapping*.json; do
        # echo "$f"
        python convert_valentine_gt.py "$f"
    done
    rm Valentine-datasets.zip
fi


# NextIA
if [ -d "./nextia/testbedS/datalake" ] && [ "$(ls -A ./nextia/testbedS/datalake)" ]; then
    echo "the NextIA dataset was already downloaded"
else
    curl -L -o testbedS.zip https://mydisk.cs.upc.edu/s/dX3FajwWZn7rrrd/download
    unzip testbedS.zip
    rm -rf ./nextia/testbedS/datalake
    mv testbedS/datasets ./nextia/testbedS/datalake
    rm -rf testbedS
    rm testbedS.zip

    curl -L -o testbedM.zip https://mydisk.cs.upc.edu/s/niPyR4WTtxydprj/download
    unzip testbedM.zip
    rm -rf ./nextia/testbedM/datalake
    mv testbedM/datasets ./nextia/testbedM/datalake
    rm -rf testbedM
    rm testbedM.zip

    # rename some files!
    (
        cd nextia/testbedM/datalake
        mv '2020-04-16 Coronavirus Tweets.CSV' '2020-04-16_Coronavirus_Tweets.CSV'
        mv '2020-04-25 Coronavirus Tweets.CSV' '2020-04-25_Coronavirus_Tweets.CSV'
        mv '2020-04-26 Coronavirus Tweets.CSV' '2020-04-26_Coronavirus_Tweets.CSV'
        mv '2020-04-27 Coronavirus Tweets.CSV' '2020-04-27_Coronavirus_Tweets.CSV'
        mv '2020-04-29 Coronavirus Tweets.CSV' '2020-04-29_Coronavirus_Tweets.CSV'
    )
fi

# Wiki Join
if [ -d "./wikijoin/datalake" ] && [ "$(ls -A ./wikijoin/datalake)" ]; then
    echo "the wikijoin dataset was already downloaded"
else
    echo "Downloading WikiJoin"
    curl -L -o ./wikijoin/original.tar.bz2 https://zenodo.org/records/10042019/files/wiki-join-search.tar.bz2
    echo "Setting up Wiki Join"
    (
        cd wikijoin
        tar -xzf gt.jsonl.tar.gz
        # take the first 100 rows from gt file to create wikijoin-small version
        head -n 100 gt.jsonl > gt_small.jsonl
        mkdir ./datalake
        tar -xjf ./original.tar.bz2
        rm -rf datalake
        mv wiki-join-search/tables-with-headers/ datalake
        rm -r wiki-join-search
        rm -r ./original.tar.bz2
    )
    echo "WikiJoin Setup Done"
fi

# Auto Join
if [ -d "./autojoin/datalake" ] && [ "$(ls -A ./autojoin/datalake)" ]; then
    echo "the autojoin dataset was already downloaded"
else
    echo "Downloading AutoJoin"
    rm -rf ./autojoin/original
    rm -rf ./autojoin/datalake
    git clone https://github.com/Yeye-He/Auto-Join.git ./autojoin/original
    (
        cd autojoin
        mkdir -p datalake
        sh ./create_datalake.sh
        rm -rf ./original
    )
    echo "AutoJoin Setup Done"
fi

popd

# Only validate if the cache files are missing (validation is slow for opendata).
# If all cache files exist, the data is already validated and counts are confirmed.
cache_base="cache/datasets/column_similarity_search"
if [ -f "${cache_base}/opendata/valid_data.json" ] && \
   [ -f "${cache_base}/autojoin/valid_data.json" ] && \
   [ -f "${cache_base}/nextia/valid_data.json" ] && \
   [ -f "${cache_base}/wikijoin_small/valid_data.json" ]; then
    echo "All datasets already validated, skipping validation step"
else
    python benchmark_src/dataset_creation/validate_join_benchmark.py
fi
