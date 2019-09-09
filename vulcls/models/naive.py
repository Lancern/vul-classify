from typing import *

import numpy as np

from vulcls.asm import Repository
from vulcls.asm import Program
from vulcls.asm import ProgramTag
from vulcls.asm import Function

from .base import AbstractModel
from .utils import softmax
from .utils import cosine_similarity


class NaiveModel(AbstractModel):
    class NaiveModelParams:
        def __init__(self, **kwargs):
            self.sim_threshold = kwargs.get('sim_threshold', 0.6)

    def __init__(self, **kwargs):
        self._repo = None
        self._params = self.__class__.NaiveModelParams(**kwargs)

    def train(self, repo: Repository) -> None:
        # Nothing to do here.
        pass

    def _dim(self) -> int:
        return len(self._repo.tags())

    def _find_tag(self, tag: ProgramTag) -> int:
        for (i, t) in enumerate(self._repo.tags()):
            if tag == t:
                return i
        return -1

    def _predict_func(self, f: Function) -> Dict[ProgramTag, int]:
        matched_tags = dict()
        for prog in self._repo.programs():
            prog_funcs = prog.funcs()
            for pf in prog_funcs:
                sim = cosine_similarity(f.vec(), pf.vec())
                if sim >= self._params.sim_threshold:
                    func_tag = prog.tag()
                    if func_tag in matched_tags:
                        matched_tags[func_tag] += 1
                    else:
                        matched_tags[func_tag] = 1

        return matched_tags

    def predict(self, repo: Repository, target: Program) -> np.ndarray:
        self._repo = repo
        target_funcs = target.funcs()

        matched_tags = dict()
        for tf in target_funcs:
            func_tags = self._predict_func(tf)
            for (tag, count) in func_tags.items():
                if tag in matched_tags:
                    matched_tags[tag] += count
                else:
                    matched_tags[tag] = count

        result = np.zeros(self._dim())
        if len(matched_tags) == 0:
            # Random predict.
            return softmax(np.random.rand(self._dim()))
        else:
            for (t, count) in matched_tags.items():
                tag_index = self._find_tag(t)
                if tag_index == -1:
                    continue
                result[tag_index] = count
            return softmax(result)

    def serialize(self, file_name: str) -> None:
        # Nothing to do here.
        return None

    def populate(self, file_name: str) -> None:
        # Nothing to do here.
        pass


__all__ = ['NaiveModel']
