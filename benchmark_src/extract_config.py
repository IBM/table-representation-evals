# extract_config.py
import hydra
from omegaconf import DictConfig, OmegaConf
import sys
from pathlib import Path
from itertools import product

from benchmark_src.utils.framework import register_resolvers

if __name__ == "__main__":
    register_resolvers()

    cli_overrides = sys.argv[1:]

    with hydra.initialize(version_base=None, config_path="./config"):
        cfg = hydra.compose(config_name="config", overrides=cli_overrides)

        for task, values in cfg.benchmark_tasks.items():
            datasets = values.datasets
            if datasets is not None:
                print(f"TASK:{task}")
                for dataset_name in datasets:
                    print(f"DATASET:{dataset_name}")
                    # Special handling for datasets with variations
                    if dataset_name == "wikidata_books":
                        print('variation')
                        dataset_config_path = Path("./benchmark_src/config/dataset") / f"{dataset_name}.yaml"
                        if not dataset_config_path.exists():
                            print(f"Could not find dataset config path: {dataset_config_path}")
                        print(dataset_config_path)
                        # load dataset config directly to avoid re-initializing Hydra
                        try:
                            dataset_cfg = OmegaConf.load(str(dataset_config_path))
                        except Exception as e:
                            print(f"Could not load dataset config: {e}")
                            continue

                        ncols = dataset_cfg['variations']['number_of_cols']
                        naming = dataset_cfg['variations']['column_naming']

                        for nc, nm in product(ncols, naming):
                            print(f"VARIATION:{dataset_name}_{nc}cols_{nm}_names")
                    else:
                        print(f"VARIATION:{dataset_name}")
