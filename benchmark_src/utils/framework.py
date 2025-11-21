import importlib
import sys
from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf

from benchmark_src.approach_interfaces.base_interface import BaseTabularEmbeddingApproach

# hydra override keys to exclude from run ID generation
EXCLUDED_KEYS: set[str] = {"experiment"}


def generate_run_id_string(_=None) -> str:
    """
    Generates a unique Hydra override string from HydraConfig, excluding any overrides whose key is present in
    EXCLUDED_KEYS.
    """
    hc = HydraConfig.get()
    if hc is None:
        raise RuntimeError("HydraConfig is unavailable. Cannot generate run string.")

    def _key(item: str) -> str:
        return item.split("=", 1)[0].strip()

    combined = [*hc.overrides.hydra, *hc.overrides.task]
    filtered = [item for item in combined if _key(item) not in EXCLUDED_KEYS]

    return ",".join(filtered)

def _sanitize_dirname(dirname: str) -> str:
    return dirname.replace("/", "_")


def _parse_string_to_dict(raw: str) -> dict:
    """
    Converts a Hydra override string into a key–value dict.
    """
    result = {}
    for item in raw.split(","):
        if "=" not in item:
            raise ValueError(f"Invalid override item without '=': {item}")
        key, value = item.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def create_run_path(override_string: str) -> str:
    """
    Converts Hydra's raw override string into a structured run path.
    """
    if not override_string:
        raise ValueError("Empty override string provided to create_run_path.")

    overrides_dict = _parse_string_to_dict(override_string)

    # Approach components
    approach_params = [
        f"{k.split('.', 1)[1]}={v}"
        for k, v in overrides_dict.items()
        if k.startswith("approach.")
    ]

    path_components = []
    if approach_params:
        path_components.append(",".join(approach_params))

    path_components.append(overrides_dict["task"])
    path_components.append(overrides_dict["dataset_name"])

    # Final sanitization
    path_components = [_sanitize_dirname(p) for p in path_components]
    return "/".join(path_components)

def register_resolvers():
    OmegaConf.register_new_resolver("generate_run_id", generate_run_id_string)
    OmegaConf.register_new_resolver("create_run_path", create_run_path)


def get_approach_class(cfg):
    """
    Based on the hydra config parameters of the chosen approach, load the custom approach module and class.
    
    Returns:

    """
    user_approach_file_path = Path(get_original_cwd()) / Path(cfg.approach.module_path) / "approach.py"

    # Ensure the path is absolute and resolve any '..'
    user_approach_file_path = user_approach_file_path.resolve()

    #print(f"Attempting to load user approach from: {user_approach_file_path}")

    # Give the module a unique name so it doesn't conflict with other imports
    # The actual filename (e.g., 'approach') is a good choice.
    module_name_for_import = user_approach_file_path.stem # Extracts 'approach' from 'approach.py'

    spec = importlib.util.spec_from_file_location(module_name_for_import, user_approach_file_path)

    if spec is None:
        raise ImportError(f"Could not find module spec for file: {user_approach_file_path}")

    user_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name_for_import] = user_module
    spec.loader.exec_module(user_module)

    print(f"Successfully loaded module '{module_name_for_import}' from '{user_approach_file_path}'.")

    class_name = cfg["approach"]["class_name"]

    # Get the approach class from the imported module
    embedding_approach_class = getattr(user_module, class_name)

    # Make sure the approach class implements the interface
    if not issubclass(embedding_approach_class, BaseTabularEmbeddingApproach):
        raise TypeError(f"The class '{class_name}' must inherit from TabularEmbeddingApproach.")
    
    #print("Found class")
    return embedding_approach_class