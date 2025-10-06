# extract_config.py
import hydra
from omegaconf import DictConfig
import sys

from benchmark_src.utils.framework import register_resolvers

if __name__ == "__main__":
    register_resolvers()
    #print(sys.argv) # TODO: make a bit more robust?
    cli_overrides = sys.argv[1:]
    #print(cli_overrides)

    with hydra.initialize(version_base=None, config_path="./config"):
        # Compose the config
        cfg = hydra.compose(config_name="config", overrides=cli_overrides)
        #print(f"Initial config: {cfg}")

        # print datasets per tasks for later extraction
        for task, values in cfg.benchmark_tasks.items():
            datasets = values.datasets

            if datasets is not None:
                print(f"TASK:{task}")
                print(f"DATASETS:{' '.join(datasets)}")
