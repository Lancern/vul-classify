import flask

app = flask.Flask(__name__)


@app.route('/')
def index():
    return 'hello'


@app.route('/classify', methods=['POST'])
def classify():
    # TODO: Implement me.
    pass


def start_httpd():
    app.run()
    return 0


__all__ = ['start_httpd']
