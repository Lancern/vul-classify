from .base import AbstractModel

import numpy as np

from ..repr import Repository
from ..repr import Program


class NaiveModel(AbstractModel):
    def __init__(self):
        self._repo = None

    def train(self, repo: Repository) -> None:
        self._repo = repo

    def predict(self, target: Program) -> np.ndarray:
        # TODO: Implement Naive Model.
        pass
