from typing import *
import importlib.util

from .base import AbstractModel


def _load_module_file(file_name) -> Any:
    spec = importlib.util.spec_from_file_location(file_name, file_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model_object(py_file: str, model_name: str) -> AbstractModel:
    module = _load_module_file(py_file)
    model_cls = module[model_name]
    return model_cls()


__all__ = [load_model_object]
