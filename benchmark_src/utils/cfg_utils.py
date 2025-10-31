from pathlib import Path
from omegaconf import OmegaConf

def load_dataset_config(dataset_name: str):
    dataset_config_path = Path("./benchmark_src/config/dataset") / f"{dataset_name}.yaml"
    if not dataset_config_path.exists():
        print(f"Could not find dataset config path: {dataset_config_path}")

    dataset_cfg = OmegaConf.load(str(dataset_config_path))

    return dataset_cfg

def load_task_config(task_name: str):
    task_config_path = Path("./benchmark_src/config/task") / f"{task_name}.yaml"
    if not task_config_path.exists():
        print(f"Could not find task config path: {task_config_path}")

    task_cfg = OmegaConf.load(str(task_config_path))

    return task_cfg