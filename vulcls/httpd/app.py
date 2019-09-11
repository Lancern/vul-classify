import logging
import os
import tempfile

import flask

from vulcls.config import app_config

from vulcls.asm import disassemble_fp
from vulcls.asm import from_asm_file_fp
from vulcls.asm import get_global_repo

from vulcls.models import get_root_model


app = flask.Flask(__name__)


# POST: /classify?file=${file_path}
@app.route('/classify', methods=['POST'])
def classify():
    file_path = flask.request.args.get('file', type=str)
    if file_path is None:
        # 400 Bad Request
        return flask.Response(status=400)

    # Generate a temporary file to hold the disassembled assembly code.
    with tempfile.TemporaryFile(mode='w+') as asm_file_fp:
        logging.debug('Disassembling file "%s" into "%s"', file_path, asm_file_fp.name)

        # noinspection PyBroadException
        try:
            disassemble_fp(file_path, asm_file_fp)
        except Exception as ex:
            # 422 Unprocessable Entity.
            logging.error('Cannot disassemble binary file "%s" into "%s": %s', file_path, asm_file_fp.name, ex)
            return flask.Response(status=422)

        # Load the assembly file to a program.
        logging.debug('Load disassembled file')
        asm_file_fp.seek(0)

        try:
            target = from_asm_file_fp(asm_file_fp, os.path.basename(file_path))
        except Exception as ex:
            # 500 Internal Server Error.
            logging.error('Failed to load Program object from assembly file, binary file is "%s"', file_path)
            raise

    # Evaluate using model.
    logging.debug('Evaluate loaded program')
    prediction_v = get_root_model().predict(get_global_repo(), target)

    return dict(
        zip(
            map(lambda t: t.label(), get_global_repo().tags()),
            prediction_v
        )
    )


def start_httpd():
    address = app_config().get('daemon.address', '127.0.0.1')
    port = app_config().get('daemon.port', 8080)

    app.run(host=address, port=port)
    return 0


__all__ = ['start_httpd']
