from typing import *
import importlib.util

from .base import AbstractModel


def _load_module_file(file_name: str, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, file_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model_object(py_file: str, module_name: str, model_name: str) -> AbstractModel:
    module = _load_module_file(py_file, module_name)
    model_cls = getattr(module, model_name)
    return model_cls()


__all__ = ['load_model_object']
