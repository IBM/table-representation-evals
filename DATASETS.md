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
https://onedrive.live.com/?id=86FDED4BAA4DCE9D%2119255&cid=86FDED4BAA4DCE9D&redeem=aHR0cHM6Ly8xZHJ2Lm1zL2YvcyFBcDNPVGFwTDdmMkdnWlkzbXJGOG5tSHYxN1RSeWc%5FZT1rWkU0T3g
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
