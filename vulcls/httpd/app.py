import flask

app = flask.Flask(__name__)


@app.route('/')
def index():
    return 'hello'


@app.route('/classify', methods=['POST'])
def classify():
    file_path = bytes.decode(flask.request.data)
    with open(file_path, 'rb') as fp:
        program_text = fp.read()

    # print(program)
    # TODO: Implement classify request handler.


def start_httpd():
    app.run()
    return 0


__all__ = ['start_httpd']
