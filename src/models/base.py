from typing import *

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

    def serialize(self) -> Any:
        # Override this method in derived classes to serialize the internal state of the model
        # into primitive values.
        pass

    def populate(self, rep: Any) -> None:
        # Override this method in derived classes to deserialize the internal state of the model
        # from primitive values.
        pass
