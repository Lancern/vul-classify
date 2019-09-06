import sys
sys.path.append('D:\\Documents\\Algorithm\\VSCode\\vul-classify')

from flask import Flask, escape, request
import vul_classify.repr
from xgb_model import XGBModel
from models import WeightedMajorityVoting

app = Flask(__name__)
xgb = XGBModel()
wmv = WeightedMajorityVoting([xgb])

@app.route('/')
def index():
    return 'hello'

@app.route('/classify', methods=['POST'])
def classify():
    file_path = bytes.decode(request.data)
    program = open(file_path, 'rb').read()

    # print(program)
    # TODO: change to repr.Program

    return wmv.predict(program)

if __name__ == "__main__":
    app.run()
