# Dataset Setup

## ECB

```
https://zenodo.org/records/10042019/files/ecb-join.tar.bz2?download=1
```

Extract to `cache/dataset_creation_resources/tables/`.

## CKAN

```
https://zenodo.org/records/10042019/files/ckan-subset.tar.bz2?download=1
```

Extract to `cache/dataset_creation_resources/ckan-subset/tables/`.

## WDC Schema.org (TTD)

```
https://github.com/awslabs/hypergraph-tabular-lm/blob/main/checkpoints/readme.txt
```

Extract to `cache/dataset_creation_resources/ttd/`.

## Final directory structure

```
cache/dataset_creation_resources/
├── tables/                    # ECB: *.csv.gz
├── ckan-subset/
│   └── tables/                # CKAN: *.csv.bz2
└── ttd/
    ├── train/                 # WDC: *.json.gz
    ├── dev/                   # WDC: *.json.gz
    └── test/                  # WDC: *.json.gz
```
