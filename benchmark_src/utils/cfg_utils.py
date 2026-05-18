from pathlib import Path
from omegaconf import OmegaConf, DictConfig, ListConfig

def load_dataset_config(dataset_name: str):
    dataset_config_path = Path("./benchmark_src/config/dataset") / f"{dataset_name}.yaml"
    if not dataset_config_path.exists():
        print(f"Could not find dataset config path: {dataset_config_path}")

    dataset_cfg = OmegaConf.load(str(dataset_config_path))

    return dataset_cfg


def _this_file_dir() -> Path:
    try:
        # Works for scripts / imports
        return Path(__file__).resolve().parent
    except NameError:
        # Fallback for Jupyter (rare but safe)
        return Path.cwd().resolve()


def load_task_config(task_name: str):
    utils_dir = _this_file_dir()

    # utils/ → benchmark_src/
    benchmark_src_dir = utils_dir.parent

    task_config_path = (
        benchmark_src_dir / "config" / "task" / f"{task_name}.yaml"
    )

    if not task_config_path.exists():
        raise FileNotFoundError(
            f"Could not find task config: {task_config_path}"
        )

    return OmegaConf.load(task_config_path)


def guard_cfg_no_none(cfg, path="cfg", _depth=0):
    """
    Recursively checks that no value in the config is None.

    Args:
        cfg: A DictConfig, dict, or nested structure.
        path: Internal parameter used for error reporting.
        _depth: Internal parameter tracking recursion depth (max 200).

    Raises:
        ValueError: If any value in cfg is None.
        RecursionError: If maximum config depth is exceeded.
    """
    _MAX_DEPTH = 200
    if _depth > _MAX_DEPTH:
        raise RecursionError(
            f"guard_cfg_no_none: exceeded max depth ({_MAX_DEPTH}) at '{path}'. "
            "The config may contain circular references."
        )

    allow_none_keys = {"cfg.test_case_limit", "cfg.task.train_limit", "cfg.task.test_limit"}
    if isinstance(cfg, DictConfig) or isinstance(cfg, dict):
        for key, value in cfg.items():
            guard_cfg_no_none(value, f"{path}.{key}", _depth + 1)
    elif isinstance(cfg, (ListConfig, list)):
        for idx, item in enumerate(cfg):
            guard_cfg_no_none(item, f"{path}[{idx}]", _depth + 1)
    else:
        if cfg is None and path not in allow_none_keys:
            raise ValueError(f"Configuration value at '{path}' is None!")
