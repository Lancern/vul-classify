from typing import *

import sys
import logging
import argparse

from .config import init_config
from .config import app_config
from .thread_pool import init_thread_pool as init_tp

from .models import init_root_model
from .models import load_model_object

from .httpd.app import start_httpd


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
            # Logging facility is not available here. Use print instead.
            print('Invalid logging file: {}'.format(log_file), file=sys.stderr)
            exit(-1)

        file_handler = logging.FileHandler(log_file, mode='w+')
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def init_thread_pool():
    workers = app_config().get('thread_pool.workers', 10)
    if not isinstance(workers, int) or workers <= 0:
        logging.error('Invalid number of workers: {}'.format(workers), file=sys.stderr)
        exit(-1)

    logging.info('Initializing thread pool')
    logging.debug('max_workers of thread pool is %d', workers)

    init_tp(workers)


def init_models():
    logging.info('Loading models')

    models_to_load = app_config().get('models', [])
    if len(models_to_load) == 0:
        logging.error('No models specified in configuration.')
        exit(-1)

    models_loaded = []
    for model in models_to_load:
        model_file = model['file']
        model_name = model['name']

        logging.debug('Loading model "%s" from file "%s"', model_name, model_file)

        model = load_model_object(model_file, model_name)
        models_loaded.append(model)

    logging.debug('%d models loaded.', len(models_loaded))
    logging.debug('Initializing root model')

    init_root_model(models_loaded)


def startup(argv: List[str]) -> None:
    parser = init_arg_parser()
    args = parser.parse_args(argv)

    # Initialize application configuration.
    init_config(args.config_file)

    # Initialize logging facilities.
    init_logging()

    # Initialize thread pool.
    init_thread_pool()

    # Initialize models.
    init_models()


def main(argv: List[str]) -> int:
    # System startup
    startup(argv)

    # Start the HTTP daemon.
    start_httpd()

    return 0


exit(main(sys.argv))
