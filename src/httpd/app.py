import flask

app = flask.Flask(__name__)


@app.route('/')
def index():
    return 'hello'


@app.route('/classify', methods=['POST'])
def classify():
    file_path = bytes.decode(flask.request.args['path'])
    with open(file_path, 'rb') as fp:
        program_text = fp.read()

    # TODO: Implement classify request handler.


def start_httpd():
    app.run()
