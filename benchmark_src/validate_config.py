#!/usr/bin/env python3
"""
Validate Hydra configuration files for the benchmark.
This script checks for common configuration issues and validates structure.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import sys
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from benchmark_src.utils.framework import register_resolvers

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConfigValidator:
    """Validates Hydra configuration files."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def validate_experiment_config(self, experiment_name: str) -> bool:
        """
        Validate an experiment configuration file.
        
        Args:
            experiment_name: Name of the experiment (without .yaml extension)
            
        Returns:
            True if valid, False otherwise
        """
        try:
            with hydra.initialize(version_base=None, config_path="./config"):
                cfg = hydra.compose(config_name="config", overrides=[f"experiment={experiment_name}"])
                return self._validate_config(cfg, experiment_name)
        except Exception as e:
            self.errors.append(f"Failed to load experiment '{experiment_name}': {e}")
            return False
    
    def _validate_config(self, cfg: DictConfig, experiment_name: str) -> bool:
        """Validate a loaded configuration."""
        is_valid = True
        
        # Check required fields
        required_fields = ['benchmark_output_dir', 'benchmark_datasets_dir', 'benchmark_tasks']
        for field in required_fields:
            if not hasattr(cfg, field) or not getattr(cfg, field):
                self.errors.append(f"Missing required field: {field}")
                is_valid = False
        
        # Validate benchmark_tasks
        if hasattr(cfg, 'benchmark_tasks') and cfg.benchmark_tasks:
            is_valid &= self._validate_benchmark_tasks(cfg.benchmark_tasks)
        
        # Validate Hydra configuration
        if hasattr(cfg, 'hydra'):
            is_valid &= self._validate_hydra_config(cfg.hydra)
        
        # Validate approach configuration
        if hasattr(cfg, 'approach'):
            is_valid &= self._validate_approach_config(cfg.approach)
        
        return is_valid
    
    def _validate_benchmark_tasks(self, tasks: DictConfig) -> bool:
        """Validate benchmark tasks configuration."""
        is_valid = True
        valid_task_types = [
            'row_similarity_search', 'predictive_ml', 'column_similarity_search',
            'more_similar_than', 'clustering'
        ]
        
        for task_name, task_config in tasks.items():
            if task_name not in valid_task_types:
                self.warnings.append(f"Unknown task type: {task_name}")
            
            # Check if task has datasets
            if not hasattr(task_config, 'datasets') or not task_config.datasets:
                self.warnings.append(f"Task '{task_name}' has no datasets defined")
            else:
                logger.info(f"Task '{task_name}' has {len(task_config.datasets)} datasets")
        
        return is_valid
    
    def _validate_hydra_config(self, hydra_config: DictConfig) -> bool:
        """Validate Hydra configuration."""
        is_valid = True
        
        # Check launcher configuration
        if hasattr(hydra_config, 'launcher'):
            launcher = hydra_config.launcher
            if hasattr(launcher, 'n_jobs') and launcher.n_jobs <= 0:
                self.errors.append("launcher.n_jobs must be positive")
                is_valid = False
            
            if hasattr(launcher, 'backend'):
                valid_backends = ['loky', 'multiprocessing', 'threading']
                if launcher.backend not in valid_backends:
                    self.warnings.append(f"Unknown launcher backend: {launcher.backend}")
        
        return is_valid
    
    def _validate_approach_config(self, approach_config: DictConfig) -> bool:
        """Validate approach configuration."""
        is_valid = True
        
        # Check required approach fields
        required_fields = ['approach_name', 'module_path', 'class_name']
        for field in required_fields:
            if not hasattr(approach_config, field) or not getattr(approach_config, field):
                self.errors.append(f"Missing required approach field: {field}")
                is_valid = False
        
        return is_valid
    
    def print_results(self) -> None:
        """Print validation results."""
        if self.errors:
            logger.error("❌ Validation errors found:")
            for error in self.errors:
                logger.error(f"  - {error}")
        
        if self.warnings:
            logger.warning("⚠️  Validation warnings:")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")
        
        if not self.errors and not self.warnings:
            logger.info("✅ Configuration is valid!")
        elif not self.errors:
            logger.info("✅ Configuration is valid with warnings")
        else:
            logger.error("❌ Configuration has errors")

def main():
    """Main function to validate configurations."""
    if len(sys.argv) < 2:
        print("Usage: python validate_config.py <experiment_name>")
        print("Example: python validate_config.py hytrel_test")
        sys.exit(1)
    
    experiment_name = sys.argv[1]
    
    # Register custom resolvers
    register_resolvers()
    
    # Validate configuration
    validator = ConfigValidator()
    is_valid = validator.validate_experiment_config(experiment_name)
    
    # Print results
    validator.print_results()
    
    # Exit with appropriate code
    sys.exit(0 if is_valid else 1)

if __name__ == "__main__":
    main()
