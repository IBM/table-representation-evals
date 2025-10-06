#!/usr/bin/env python3
"""
Extract benchmark configuration from Hydra config files.
This script parses experiment configurations and outputs task/dataset combinations.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import sys
import logging
from typing import List, Dict, Any

from benchmark_src.utils.framework import register_resolvers

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_benchmark_config(cli_overrides: List[str]) -> Dict[str, List[str]]:
    """
    Extract benchmark configuration from Hydra config.
    
    Args:
        cli_overrides: Command line overrides for Hydra
        
    Returns:
        Dictionary mapping task names to list of datasets
    """
    try:
        with hydra.initialize(version_base=None, config_path="./config"):
            # Compose the config
            cfg = hydra.compose(config_name="config", overrides=cli_overrides)
            
            # Validate configuration
            if not hasattr(cfg, 'benchmark_tasks') or not cfg.benchmark_tasks:
                raise ValueError("No benchmark_tasks found in configuration")
            
            # Extract task/dataset combinations
            task_datasets = {}
            for task_name, task_config in cfg.benchmark_tasks.items():
                if hasattr(task_config, 'datasets') and task_config.datasets:
                    task_datasets[task_name] = list(task_config.datasets)
                    logger.info(f"Found task '{task_name}' with {len(task_config.datasets)} datasets")
                else:
                    logger.warning(f"Task '{task_name}' has no datasets defined")
            
            return task_datasets
            
    except Exception as e:
        logger.error(f"Failed to extract configuration: {e}")
        raise

def print_task_datasets(task_datasets: Dict[str, List[str]]) -> None:
    """
    Print task/dataset combinations in the expected format.
    
    Args:
        task_datasets: Dictionary mapping task names to list of datasets
    """
    for task_name, datasets in task_datasets.items():
        print(f"TASK:{task_name}")
        print(f"DATASETS:{' '.join(datasets)}")

def main():
    """Main function to extract and print benchmark configuration."""
    try:
        # Register custom resolvers
        register_resolvers()
        
        # Parse command line arguments
        cli_overrides = sys.argv[1:]
        if not cli_overrides:
            logger.error("No command line overrides provided")
            sys.exit(1)
        
        logger.info(f"Processing overrides: {cli_overrides}")
        
        # Extract configuration
        task_datasets = extract_benchmark_config(cli_overrides)
        
        if not task_datasets:
            logger.error("No valid task/dataset combinations found")
            sys.exit(1)
        
        # Print results
        print_task_datasets(task_datasets)
        
        logger.info("Configuration extraction completed successfully")
        
    except Exception as e:
        logger.error(f"Configuration extraction failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
