#!/bin/bash
set -e 

pushd .

cd ContextAwareJoin/datasets/

#bash download.sh # (downlaods all datasets)

# Open Data
echo "Downloading OpenData"
wget https://zenodo.org/records/15881731/files/opendata-contextawarejoins.zip
unzip opendata-contextawarejoins.zip 
mv opendata-contextawarejoins opendata
rm opendata-contextawarejoins.zip 
echo "OpenData Setup Done"

# Valentine
wget -O Valentine-datasets.zip "https://zenodo.org/records/5084605/files/Valentine-datasets.zip?download=1"
unzip Valentine-datasets.zip
rm -rf ./valentine
mkdir ./valentine
mv Valentine-datasets ./valentine
for f in ./valentine/Valentine-datasets/*/Semantically-Joinable/*/*mapping*.json;do `python convert_valentine_gt.py $f`;done
rm Valentine-datasets.zip

popd

