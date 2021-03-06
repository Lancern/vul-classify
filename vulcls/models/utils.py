from typing import *

import numpy as np


def softmax(v: np.ndarray) -> np.ndarray:
    ev = np.exp(v)
    s = np.sum(ev)
    return ev / s


def cosine_similarity(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    return np.dot(lhs, rhs) / (np.linalg.norm(lhs) * np.linalg.norm(rhs))


def select_max(v: np.ndarray, values: List[Any]) -> Any:
    return values[np.argmax(v)]


__all__ = ['softmax', 'cosine_similarity', 'select_max']
