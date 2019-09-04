from typing import *

import numpy as np

import vul_classify.repr
import vul_classify.concurrent


def softmax(v: np.ndarray) -> np.ndarray:
    ev = np.exp(v)
    s = np.sum(ev)
    return ev / s


def cosine_similarity(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    return np.dot(lhs, rhs) / (np.linalg.norm(lhs) * np.linalg.norm(rhs))


class AbstractModel:
    def train(self, repo: vul_classify.repr.Repository) -> None:
        # Override this method in derived classes.
        pass

    def predict(self, target: vul_classify.repr.Program) -> np.ndarray:
        # Override this method in derived classes.
        pass


class WeightedMajorityVoting(AbstractModel):
    def __init__(self, models: List[AbstractModel]):
        self._models = models
        self._w = np.ones(len(models))

    def train(self, repo: vul_classify.repr.Repository) -> None:
        def train_model(model: AbstractModel) -> None:
            model.train(repo)

        vul_classify.concurrent.get_thread_pool().map(train_model, *self._models)

    def predict(self, target: vul_classify.repr.Program) -> np.ndarray:
        def predict_by(model: AbstractModel) -> np.ndarray:
            return model.predict(target)

        predictions = list(vul_classify.concurrent.get_thread_pool().map(predict_by, self._models))
        y = np.vstack(predictions)
        return softmax(np.matmul(self._w, y))


class NaiveModel(AbstractModel):
    def __init__(self):
        self._repo = None

    def train(self, repo: vul_classify.repr.Repository) -> None:
        self._repo = repo

    def predict(self, target: vul_classify.repr.Program) -> np.ndarray:
        # TODO: Implement Naive Model.
        pass
