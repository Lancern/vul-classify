import logging

from vulcls.config import app_config
from vulcls.asm import get_global_repo
from vulcls.models import get_root_model


def train():
    logging.info('Process launched in train mode.')

    # Check if every loaded model has corresponding data file in configuration.
    model_data_files = dict()
    for model_info in app_config().get('models', []):
        if 'data_file' not in model_info:
            logging.error('"data_file" property not found in configuration of model "%s"', model_info['name'])
            return -1
        model_data_files[model_info['name']] = model_info['data_file']

    logging.info('Training')

    get_root_model().train(get_global_repo())

    logging.info('Training complete.')
    logging.info('Serializing models to local files')

    for model in get_root_model().underlying_models():
        data_file = model_data_files[model.__class__.__name__]
        logging.info('Serializing %s to data file "%s"', model.__class__.__name__, data_file)
        model.serialize(data_file)

    return 0


__all__ = ['train']
