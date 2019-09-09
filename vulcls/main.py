import sys
import logging
import argparse

from vulcls.train import train

from vulcls.config import init_config
from vulcls.config import app_config
from vulcls.thread_pool import init_thread_pool as init_tp

from vulcls.asm import set_global_repo
from vulcls.asm import get_global_repo
from vulcls.asm import deserialize_repo

from vulcls.models import init_root_model
from vulcls.models import load_model_object

from vulcls.httpd.app import start_httpd


if __name__ != '__main__':
    raise Exception('main.py should not be imported as a module.')


def init_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Model daemon for vul-classify Viper plugin.')
    parser.add_argument('-c', '--config', type=str, required=False, default='vul-classify.config.json', metavar='path',
                        dest='config_file', help='specify the path of the configuration file.')
    parser.add_argument('-t', '--train', action='store_true', dest='train',
                        help='launch the process in train mode.')

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


def init_repo():
    repo_file_name = app_config().get('repo.filename', None)
    if repo_file_name is None:
        logging.error('No repository file found in configuration.')
        exit(-1)

    logging.info('Loading global repository from file "%s"', repo_file_name)
    set_global_repo(deserialize_repo(repo_file_name))

    logging.debug('%d programs loaded from repository', len(get_global_repo().programs()))


def init_models():
    logging.info('Loading models')

    models_to_load = app_config().get('models', [])
    if len(models_to_load) == 0:
        logging.error('No models specified in configuration.')
        exit(-1)

    models_loaded = []
    for model in models_to_load:
        model_file = model['file']
        model_module = model['module']
        model_name = model['name']

        logging.debug('Loading model "%s" from file "%s"', model_name, model_file)

        model = load_model_object(model_file, model_module, model_name)
        models_loaded.append(model)

    logging.debug('%d models loaded.', len(models_loaded))
    logging.debug('Initializing root model')

    init_root_model(models_loaded)


def startup() -> bool:
    parser = init_arg_parser()
    args = parser.parse_args()

    # Initialize application configuration.
    init_config(args.config_file)

    # Initialize logging facilities.
    init_logging()

    # Initialize thread pool.
    init_thread_pool()

    # Initialize repository.
    init_repo()

    # Initialize models.
    init_models()

    return not args.train


def main() -> int:
    # System startup
    if startup():
        # Start the HTTP daemon.
        return start_httpd()
    else:
        # The process is launched in train mode.
        return train()


exit(main())
