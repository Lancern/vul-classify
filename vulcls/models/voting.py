from typing import *

import numpy as np

from vulcls.asm import Repository
from vulcls.asm import Program
from vulcls.thread_pool import get_thread_pool

from .base import AbstractModel
from .utils import softmax


class WeightedMajorityVoting(AbstractModel):
    def __init__(self, models: List[AbstractModel]):
        self._models = models
        self._w = np.ones(len(models))

    def train(self, repo: Repository) -> None:
        def train_model(model: AbstractModel) -> None:
            model.train(repo)

        get_thread_pool().map(train_model, *self._models)

    def predict(self, repo: Repository, target: Program) -> np.ndarray:
        def predict_by(model: AbstractModel) -> np.ndarray:
            return model.predict(repo, target)

        predictions = list(get_thread_pool().map(predict_by, self._models))
        y = np.vstack(predictions)
        return softmax(np.matmul(self._w, y))

    def serialize(self, file_name: str) -> None:
        # Nothing to do here.
        return None

    def deserialize(self, file_name: str) -> None:
        # Nothing to do here.
        pass


__all__ = ['WeightedMajorityVoting']
