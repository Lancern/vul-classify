import numpy as np

from ..asm import Repository
from ..asm import Program


class AbstractModel:
    def train(self, repo: Repository) -> None:
        # Override this method in derived classes.
        pass

    def predict(self, target: Program) -> np.ndarray:
        # Override this method in derived classes.
        pass
