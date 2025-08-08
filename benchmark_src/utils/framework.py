from pathlib import Path
from omegaconf import OmegaConf
from hydra.utils import get_original_cwd
import importlib
import sys

from benchmark_src.approach_interfaces.base_interface import BaseTabularEmbeddingApproach

def sanitize_dirname(dirname: str) -> str:
  """Sanitizes a directory name by replacing slashes with underscores."""
  return dirname.replace('/', '_')

def register_resolvers():
    # Register the resolver
    OmegaConf.register_new_resolver('sanitize_dirname', sanitize_dirname)

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