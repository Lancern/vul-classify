from typing import *
import logging

import umsgpack

import asm2vec.model


_model: Optional[asm2vec.model.Asm2Vec] = None


def init_asm2vec(model_file_name: str) -> None:
    logging.info('Initializing asm2vec from file "%s"', model_file_name)

    memento = asm2vec.model.Asm2VecMemento()
    with open(model_file_name, 'rb') as fp:
        memento_data = umsgpack.unpack(fp)
    memento.populate(memento_data)

    global _model
    _model = asm2vec.model.Asm2Vec()
    _model.set_memento(memento)


def get_asm2vec() -> asm2vec.model.Asm2Vec:
    return _model


__all__ = ['init_asm2vec', 'get_asm2vec']
