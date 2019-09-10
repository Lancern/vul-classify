from typing import *
import logging
import os
import random

import asm2vec.logging

from vulcls.asm import Program
from vulcls.asm import ProgramTag
from vulcls.asm import get_global_repo
from vulcls.asm import from_asm_file

from vulcls.models import NaiveModel
from vulcls.models.utils import select_max


def load_program_labels() -> Dict[str, int]:
    program_label_file = os.getenv('PROGRAM_LABELS_FILE')
    if program_label_file is None:
        logging.error('PROGRAM_LABELS_FILE not set.')
        raise RuntimeError('PROGRAM_LABELS_FILE not set.')

    with open(program_label_file, 'r') as fp:
        return dict(
            map(
                lambda ln: (
                    ln[0],
                    int(ln[1])
                ),
                map(lambda t: t.strip().split(','), fp.readlines()[1:])
            )
        )


def load_test_set() -> List[Tuple[Program, ProgramTag]]:
    # Load program labels.
    program_labels = load_program_labels()

    logging.info('Program labels loaded.')

    # Reduce test set to 5%.
    reduced_program_labels = dict()
    while len(reduced_program_labels) == 0:
        for (program_name, label) in program_labels.items():
            if random.random() <= 5 / 100:
                reduced_program_labels[program_name] = label

    logging.info('%d programs selected in test set.', len(reduced_program_labels))

    asm_dir = os.getenv('ASM_FILE_DIR')
    if asm_dir is None:
        logging.error('ASM_FILE_DIR is not set.')
        raise RuntimeError('ASM_FILE_DIR is not set.')

    test_set = []
    progress = 1
    for (program_name, label) in reduced_program_labels.items():
        program_path = os.path.join(asm_dir, program_name + '.txt')
        program = from_asm_file(program_path)
        test_set.append((program, ProgramTag(label)))

        logging.debug('Program "%s" parsed, progress %f%%', program_name, progress / len(reduced_program_labels) * 100)
        progress += 1

    return test_set


def load_counters(test_set: List[Tuple[Program, ProgramTag]]) -> Dict[int, Dict[int, int]]:
    return dict(
        map(
            lambda t: (
                t[1].label(),
                dict(
                    map(
                        lambda p: (
                            p[1].label(),
                            0
                        ),
                        test_set
                    )
                )
            ),
            test_set
        )
    )


def naive_baseline():
    # Disable logging in asm2vec.
    asm2vec.logging.config_asm2vec_logging(level=logging.INFO)

    test_set = load_test_set()
    logging.debug('%d programs loaded from test set', len(test_set))

    model = NaiveModel()
    counters = load_counters(test_set)

    progress = 1
    for (program, actual_tag) in test_set:
        predict = model.predict(get_global_repo(), program)
        predict_tag = select_max(predict, get_global_repo().tags())

        counters[actual_tag.label()][predict_tag.label()] += 1

        logging.debug('Program "%s" evaluated. Progress: %f%%', program.name(), progress / len(test_set) * 100)
        progress += 1

    logging.info('Baseline test for Naive Model complete.')
    logging.info('Saving baseline result to file "naive-baseline-result.csv"')
    with open('naive-baseline-result.csv', 'w') as fp:
        fp.write(','.join(map(str, counters.keys())) + '\n')
        for c in counters.values():
            fp.write(','.join(map(str, c.values())) + '\n')

    logging.info('Done')


__all__ = ['naive_baseline']
