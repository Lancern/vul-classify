from typing import *

import numpy as np


class Function:
    def __init__(self, func_id: int, v: np.ndarray):
        self._id = func_id
        self._callees = []
        self._callers = []
        self._v = v

    def id(self) -> int:
        return self._id

    def callees(self) -> List['Function']:
        return self._callees

    def callers(self) -> List['Function']:
        return self._callers

    def vec(self) -> np.ndarray:
        return self._v


class ProgramTag:
    def __init__(self, vul_cat: str = None):
        self._cat = vul_cat

    def __eq__(self, other):
        if not isinstance(other, ProgramTag):
            return False
        return self._cat == other._cat

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self._cat)

    def is_vul(self) -> bool:
        return self._cat is not None

    def vul_category(self) -> str:
        return self._cat


class Program:
    def __init__(self, name: str, entry: Function):
        self._name = name
        self._tag = ProgramTag()
        self._entry = entry

    def name(self) -> str:
        return self._name

    def tag(self) -> ProgramTag:
        return self._tag

    def set_tag(self, tag: ProgramTag) -> None:
        self._tag = tag

    def entry(self) -> Function:
        return self._entry


class Repository:
    def __init__(self):
        self._programs = []

    def add_program(self, p: Program) -> None:
        self._programs.append(p)

    def tags(self) -> List[ProgramTag]:
        # Use dict instead of set to remove duplicate tags since set is unordered while dict is ordered.
        return list(
            dict(
                zip(
                    map(lambda prog: prog.tag(), self._programs),
                    [None] * len(self._programs)
                )
            ).keys())

    def programs(self) -> List[Program]:
        return self._programs


def walk_functions(entry: Function, handler: Callable[[Function], Any]) -> None:
    visited_funcs = set()

    def _walk(curr: Function) -> None:
        if curr.id() in visited_funcs:
            return

        visited_funcs.add(curr.id())
        handler(curr)

        for next_func in curr.callees():
            _walk(next_func)

    _walk(entry)


def collect_functions(entry: Function) -> List[Function]:
    fn = []

    def handler(f: Function) -> None:
        fn.append(f)

    walk_functions(entry, handler)
    return fn


__all__ = [Function, ProgramTag, Program, Repository, walk_functions, collect_functions]
