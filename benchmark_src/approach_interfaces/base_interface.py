import pandas as pd
from omegaconf import DictConfig
from abc import ABC, abstractmethod
from pathlib import Path
import importlib
import sys
from hydra.utils import get_original_cwd

class BaseTabularEmbeddingApproach(ABC):

    def __init__(self, cfg: DictConfig):
        self.approach_name = cfg.approach.approach_name
        self.cfg = cfg

        self._approach_path = Path(get_original_cwd()) / Path(cfg.approach.module_path)

        self._loaded_components = {} 

    # --- Methods that you want to re-use in multiple components  ---

    def train_model_self_supervised(self, your_custom_parameters=None):
        """
        If your approach is trained / adapted to the input table in a self-supervised way, 
        we recommend you to implement it here and call the method from the task-specific components.

        """
        # --- YOUR IMPLEMENTATION GOES HERE ---
        pass
    
    # --- Methods for component functionality  ---

    def _load_component(self, module_file: str, class_name: str, interface: type):
        """
        Description
        """
        try:
            module_path = self._approach_path / f"{module_file}.py"
            spec = importlib.util.spec_from_file_location(module_file, module_path)
            if spec is None:
                print(f"Component {class_name} is not implemented by appoach.")
                raise NotImplementedError
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_file] = module
            spec.loader.exec_module(module)

            ComponentClass = getattr(module, class_name)
            component_instance = ComponentClass(approach_instance=self)

            if not isinstance(component_instance, interface):
                raise TypeError(f"Component '{class_name}' does not implement '{interface.__name__}'")

            print(f"BaseEmbeddingApproach: Loaded component: {class_name}")
            self._loaded_components[module_file] = component_instance

            return component_instance
        except (ImportError, AttributeError, FileNotFoundError) as e:
            print(f"BaseEmbeddingApproach: Error loading component '{module_file}': {e}")
        except TypeError as e:
            print(f"BaseEmbeddingApproach: Component validation failed for '{class_name}': {e}")

