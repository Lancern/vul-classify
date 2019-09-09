from typing import *

import umsgpack

import numpy as np


class Function:
    def __init__(self, func_id: int, name: str, v: np.ndarray):
        self._id = func_id
        self._name = name
        self._callees = []
        self._callers = []
        self._v = v

    def id(self) -> int:
        return self._id

    def name(self) -> str:
        return self._name

    def callees(self) -> List['Function']:
        return self._callees

    def add_callee(self, callee: 'Function') -> None:
        self._callees.append(callee)
        callee._callers.append(self)

    def callers(self) -> List['Function']:
        return self._callers

    def add_caller(self, caller: 'Function') -> None:
        self._callers.append(caller)
        caller._callees.append(self)

    def vec(self) -> np.ndarray:
        return self._v


class ProgramTag:
    def __init__(self, label: int):
        self._label = label

    def __eq__(self, other):
        if not isinstance(other, ProgramTag):
            return False
        return self._label == other._label

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self._label)

    def label(self) -> int:
        return self._label


class Program:
    def __init__(self, name: str, tag: ProgramTag = None):
        self._name = name
        self._tag = tag
        self._entries = []

    def name(self) -> str:
        return self._name

    def tag(self) -> ProgramTag:
        return self._tag

    def entries(self) -> List[Function]:
        return self._entries

    def add_entry(self, f: Function) -> None:
        self._entries.append(f)

    def funcs(self) -> List[Function]:
        reachable = []
        visited_funcs = set(map(lambda f: f.id(), self._entries))

        fn = self._entries[:]
        while len(fn) > 0:
            current_fn = fn.pop()
            reachable.append(current_fn)
            for callee_fn in current_fn.callees():
                if callee_fn.id() in visited_funcs:
                    continue
                visited_funcs.add(callee_fn.id())
                fn.append(callee_fn)

        return reachable


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


_global_repo: Optional[Repository] = None


def set_global_repo(repo: Repository):
    global _global_repo
    _global_repo = repo


def get_global_repo() -> Repository:
    return _global_repo


def deserialize_repo(filename: str) -> Repository:
    with open(filename, 'rb') as fp:
        repo_data = umsgpack.unpack(fp)

    func_callees = dict(map(lambda x: (x['id'], x['callees']), repo_data['funcs'].values()))
    funcs = dict(map(lambda x: (x['id'], Function(x['id'], x['name'], np.array(x['vec']))), repo_data['funcs'].values()))
    for fn in funcs:
        for callee_id in func_callees[fn.id()]:
            fn.add_callee(funcs[callee_id])

    repo = Repository()
    for program_data in repo_data['programs']:
        program = Program(program_data['name'], ProgramTag(program_data['label']))
        for fid in program_data['entries']:
            program.add_entry(funcs[fid])

        repo.add_program(program)

    return repo


__all__ = ['Function', 'ProgramTag', 'Program', 'Repository', 'set_global_repo', 'get_global_repo', 'deserialize_repo']
