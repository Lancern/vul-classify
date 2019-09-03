from typing import *

import numpy as np

import vul_classify.repr


def softmax(v: np.ndarray) -> np.ndarray:
    ev = np.exp(v)
    s = np.sum(ev)
    return ev / s


class AbstractModel:
    def train(self, repo: vul_classify.repr.Repository) -> None:
        # Override this method in derived classes.
        pass

    def predict(self, target: vul_classify.repr.Program) -> np.ndarray:
        pass


class WeightedMajorityVoting(AbstractModel):
    def __init__(self, models: List[AbstractModel]):
        self._models = models
        self._w = np.ones(len(models))

    def train(self, repo: vul_classify.repr.Repository) -> None:
        # TODO: Concurrency training
        pass

    def predict(self, target: vul_classify.repr.Program) -> np.ndarray:
        # TODO: Concurrency prediction
        y = np.vstack(list(map(lambda m: m.predict(target), self._models)))
        return softmax(np.matmul(self._w, y))


class NaiveModel(AbstractModel):
    def __init__(self):
        self._repo = None

    def train(self, repo: vul_classify.repr.Repository) -> None:
        self._repo = repo

    def predict(self, target: vul_classify.repr.Program) -> np.ndarray:
        pass
