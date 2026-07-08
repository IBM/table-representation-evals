from pathlib import Path
from omegaconf import OmegaConf

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_dataset_config(dataset_name: str):
    dataset_config_path = _PROJECT_ROOT / "configs" / "dataset" / f"{dataset_name}.yaml"
    if not dataset_config_path.exists():
        print(f"Could not find dataset config path: {dataset_config_path}")

    dataset_cfg = OmegaConf.load(str(dataset_config_path))

    return dataset_cfg


def load_task_config(task_name: str):
    task_config_path = _PROJECT_ROOT / "configs" / "task" / f"{task_name}.yaml"

    if not task_config_path.exists():
        raise FileNotFoundError(
            f"Could not find task config: {task_config_path}"
        )

    return OmegaConf.load(task_config_path)


def load_metric_ranges() -> dict:
    """(min, max) axis domain per metric, from configs/metric_information.yaml."""
    metric_info_path = _PROJECT_ROOT / "configs" / "metric_information.yaml"
    cfg = OmegaConf.load(metric_info_path)
    return OmegaConf.to_container(cfg.metric_ranges, resolve=True)


def load_approach_plotting() -> dict[tuple[str, str], dict]:
    """
    Curated display name/color per (approach, configuration) pair, from
    configs/approach_plotting.yaml. Entries may omit "name" or "color"; callers
    should fall back for whichever field is absent.
    """
    approach_plotting_path = _PROJECT_ROOT / "configs" / "approach_plotting.yaml"
    entries = OmegaConf.to_container(OmegaConf.load(approach_plotting_path), resolve=True)
    return {(entry["approach"], entry["configuration"]): entry for entry in entries}


def guard_cfg_no_none(cfg, path="cfg"):
    """
    Recursively checks that no value in the config is None.

    Args:
        cfg: A DictConfig, dict, or nested structure.
        path: Internal parameter used for error reporting.

    Raises:
        ValueError: If any value in cfg is None.
    """
    allow_none_keys = set(["cfg.test_case_limit", "cfg.task.max_queries"])
    # approach.supported_tasks is orchestrator metadata with intentional nulls (e.g. max_queries: null)
    allow_none_prefixes = ("cfg.approach.supported_tasks", "cfg.approach.model_config")
    if any(path.startswith(p) for p in allow_none_prefixes):
        return
    if OmegaConf.is_dict(cfg) or isinstance(cfg, dict):
        for key, value in cfg.items():
            guard_cfg_no_none(value, f"{path}.{key}")
    elif OmegaConf.is_list(cfg) or isinstance(cfg, list):
        for idx, item in enumerate(cfg):
            guard_cfg_no_none(item, f"{path}[{idx}]")
    else:
        if cfg is None and path not in allow_none_keys:
            raise ValueError(f"Configuration value at '{path}' is None!")
