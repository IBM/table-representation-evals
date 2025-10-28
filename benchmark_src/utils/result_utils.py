import json
import logging

logger = logging.getLogger(__name__)


def save_results(cfg, metrics: dict):
    """
    Saves the results to disk.

        Args:
            - metrics: dictionary of task specific metrics
    """
    print(f"Dataset: {cfg.dataset_name}")

    results = {
    "task": cfg.task.task_name,
    "dataset": cfg.dataset_name,
    "approach": cfg.approach.approach_name,
    }

    results.update(metrics)

    try:
        with open("results.json", "w") as file:
            json.dump(results, file, indent=2)
    except (TypeError, OverflowError):
        logger.error(f"Received result dict that is not json serializable:")
        logger.error(f"Dict: {metrics}")

    print("Saved metrics to disk")
