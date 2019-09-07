from typing import *

import sys
import logging
import argparse

from .config import init_config
from .config import app_config
from .thread_pool import init_thread_pool as init_tp


if __name__ != '__main__':
    raise Exception('main.py should not be imported as a module.')


def init_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Model daemon for vul-classify Viper plugin.')
    parser.add_argument('-c', '--config', type=str, required=False, default='vul-classify.config.json', metavar='path',
                        dest='config_file', help='specify the path of the configuration file.')

    return parser


def init_logging():
    log_level = app_config().get('logging.level', 'INFO')

    log_levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }

    if log_level not in log_levels:
        print('Invalid log level: {}'.format(log_level), file=sys.stderr)
        exit(-1)
    log_level = log_levels[log_level]

    root = logging.getLogger()
    root.setLevel(log_level)

    formatter = logging.Formatter('[%(asctime)s %(name)s %(thread)d][%(levelname)s]: %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    log_file = app_config().get('logging.file')
    if log_file is not None:
        if not isinstance(log_file, str):
            print('Invalid logging file: {}'.format(log_file), file=sys.stderr)
            exit(-1)

        file_handler = logging.FileHandler(log_file, mode='w+')
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def init_thread_pool():
    workers = app_config().get('thread_pool.workers', 10)
    if not isinstance(workers, int) or workers <= 0:
        print('Invalid number of workers: {}'.format(workers), file=sys.stderr)
        exit(-1)

    init_tp(workers)


def main(argv: List[str]) -> int:
    parser = init_arg_parser()
    args = parser.parse_args(argv)

    # Initialize application configuration.
    init_config(args.config_file)

    # Initialize logging facilities.
    init_logging()

    # Initialize thread pool.
    init_thread_pool()

    # TODO: Implement main.

    return 0


exit(main(sys.argv))
