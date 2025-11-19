#!/bin/bash
set -e 

pushd .

cd ContextAwareJoin/datasets/

#bash download.sh # (downloads all datasets from ContextAwareJoin)

for f in *; do
    if [ -f "$f" ]; then
        echo "$f"
    fi
done

# Open Data
if [ -d "./opendata" ] && [ "$(ls -A ./opendata)" ]; then
    echo "the opendata dataset was already downloaded."
else
    echo "./opendata is empty or does not exist."
    echo "Downloading OpenData"
    wget https://zenodo.org/records/15881731/files/opendata-contextawarejoins.zip
    unzip opendata-contextawarejoins.zip 
    mv opendata-contextawarejoins opendata
    rm opendata-contextawarejoins.zip 
    echo "OpenData Setup Done"
fi

# Valentine
if [ -d "./valentine" ] && [ "$(ls -A ./valentine)" ]; then
    echo "the valentine dataset was already downloaded"
else
    echo "./valentine is empty or does not exist." 
    wget -O Valentine-datasets.zip "https://zenodo.org/records/5084605/files/Valentine-datasets.zip?download=1"
    unzip Valentine-datasets.zip
    rm -rf ./valentine
    mkdir ./valentine
    mv Valentine-datasets ./valentine
    for f in ./valentine/Valentine-datasets/*/Semantically-Joinable/*/*mapping*.json;do `python convert_valentine_gt.py $f`;done
    rm Valentine-datasets.zip
fi

popd

