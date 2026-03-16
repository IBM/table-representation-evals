# extract_config.py
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
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

            # exclude certain datasets if specified
            if "exclude_datasets" in values.keys():
                datasets_to_exclude = values.exclude_datasets
            else:
                datasets_to_exclude = []

            if datasets is not None:
                print(f"TASK:{task}")
                for dataset_name in datasets:
                    if dataset_name not in datasets_to_exclude:
                        print(f"DATASET:{dataset_name}")
                        # Special handling for datasets with variations
                        if dataset_name == "wikidata_books_variations":
                            dataset_config_path = Path("./benchmark_src/config/dataset") / f"{dataset_name}.yaml"
                            if not dataset_config_path.exists():
                                print(f"Could not find dataset config path: {dataset_config_path}")
                            #print(dataset_config_path)
                            # load dataset config directly to avoid re-initializing Hydra
                            try:
                                dataset_cfg = OmegaConf.load(str(dataset_config_path))
                            except Exception as e:
                                print(f"Could not load dataset config: {e}")
                                continue
                            
                            variations = dataset_cfg['variations']

                            for variation_name, variation_cfg in variations.items():
                                # Make sure all variation parameters are lists
                                param_lists = {k: list(v) if isinstance(v, ListConfig) else ([v] if not isinstance(v, list) else v) for k, v in variation_cfg.items()}

                                # Get cartesian product of all parameter values
                                for param_values in product(*param_lists.values()):
                                    param_combo = dict(zip(param_lists.keys(), param_values))

                                    # Create a variation identifier string
                                    variation_id = "---".join(f"{k}::{v}" for k, v in param_combo.items())
                                    print(f"VARIATION: {dataset_name}@@{variation_id}")
                        else:
                            print(f"VARIATION:{dataset_name}")
