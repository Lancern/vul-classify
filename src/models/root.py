from typing import *

from .base import AbstractModel
from .voting import WeightedMajorityVoting


_root: Optional[WeightedMajorityVoting] = None


def init_root_model(sub_models: List[AbstractModel]) -> None:
    global _root
    _root = WeightedMajorityVoting(sub_models)


def get_root_model():
    return _root


__all__ = [init_root_model, get_root_model]
