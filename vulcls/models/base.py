import numpy as np

from vulcls.asm import Repository
from vulcls.asm import Program


class AbstractModel:
    def train(self, repo: Repository) -> None:
        # Override this method in derived classes.
        pass

    def predict(self, repo: Repository, target: Program) -> np.ndarray:
        # Override this method in derived classes.
        pass

    def serialize(self, file_name: str) -> None:
        # Override this method in derived classes to serialize the internal state of the model
        # into the given file.
        pass

    def populate(self, file_name: str) -> None:
        # Override this method in derived classes to deserialize the internal state of the model
        # from the given file.
        pass


__all__ = ['AbstractModel']
