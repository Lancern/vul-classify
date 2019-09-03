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

    def predict(self, repo: vul_classify.repr.Repository) -> np.ndarray:
        pass


class WeightedMajorityVoting(AbstractModel):
    def __init__(self, models: List[AbstractModel]):
        self._models = models
        self._w = np.ones(len(models))

    def train(self, repo: vul_classify.repr.Repository) -> None:
        # TODO: Concurrency training
        pass

    def predict(self, repo: vul_classify.repr.Repository) -> np.ndarray:
        # TODO: Concurrency prediction
        y = np.vstack(list(map(lambda m: m.predict(repo), self._models)))
        return softmax(np.matmul(self._w, y))
