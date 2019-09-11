import sys
sys.path.append("..")
import math

import numpy as np

import torch
import torch.nn as nn
from torch import optim
import pdb
import umsgpack
import pandas

from base import AbstractModel
from vulcls.asm import Repository
from vulcls.asm import Program
from vulcls.asm import Function
from vulcls.asm import ProgramTag


def transform(prog, precesion=.01):
    functions = {}
    queue = prog.entries()
    for func in queue:
        functions[func.id()] = func
    start, end = 0, len(queue)
    while start < end:
        start_func = queue[start]
        start += 1
        for func in start_func.callees():
            if func.id() not in functions:
                queue.append(func)
                end += 1
                functions[func.id()] = func
    num = len(functions.keys())
    convert = {v:k for k, v in enumerate(functions.keys())}
    reconvert = {v:k for k, v in convert.items()}
    weight_feature = np.array([1. for _ in range(num)])
    for _ in range(10):
        count = 0
        for i, funcs in enumerate(functions.values()):
            temp = 0
            for func in funcs.callers():
                temp += weight_feature[convert[func.id()]]/len(func.callees())
            if funcs.callers() != []:
                count += temp-weight_feature[i]
                weight_feature[i] = .5*temp+weight_feature[i]*.5
        if count/num < precesion:
            break
    idx = np.argsort(weight_feature)[::-1]
    feature = np.array([functions[reconvert[i]].vec() for i in idx])
    return feature.reshape(-1, feature.shape[-1])


class LSTMModel(AbstractModel):

    def __init__(self, load_path=None):
        self.load_path = load_path

    def train(self, repo, batch_size=32, save_path=None, save_per_iter=5000, show_per_iters=50, iters=10000, lr=0.01, lstm_size=300, num_layers=2, keep_prob=.5):
        #pdb.set_trace()
        batch_generator = BatchGenerator(repo, batch_size)
        feat_dim = batch_generator.get_feat_dim()
        self.model = Model(batch_size, len(repo.tags()), feat_dim, lstm_size, num_layers, keep_prob)
        self.model = self.model.train()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.zero_grad()
        if self.load_path is not None:
            self.model.load(self.load_path)
        optimizer = optim.Adam(self.model.parameters())
        train_data = batch_generator.forward()
        for i in range(iters):
            optimizer.zero_grad()
            feature, seq_length, pred = next(train_data)
            feature = torch.tensor(feature, dtype=torch.float32)
            seq_length = torch.tensor(seq_length, dtype=torch.long)
            pred = torch.tensor(pred, dtype=torch.long)
            if torch.cuda.is_available():
                feature = feature.cuda()
                seq_length = seq_length.cuda()
                pred = pred.cuda()
            logits, _ = self.model(feature, seq_length)
            loss = self.model.loss(logits, pred)
            loss.backward()
            optimizer.step()
            if (i+1) % show_per_iters == 0:
                print("iters:{:}    loss: {:.2f}".format(i+1, loss))
            if (i + 1) % save_per_iter == 0:
                self.model.save(save_path.split('.')[0]+"_"+str(i)+"."+save_path.split('.')[1])
            torch.cuda.empty_cache()

    def predict(self, repo, prog):
        feature = transform(prog)
        feature = feature.reshape((1,)+feature.shape)
        assert len(feature.shape) == 3 and feature.shape[0] == 1, "feature must be [1, seq_length, feat_dim]"
        self.model = Model(num_classes=len(repo.tags()), feat_dim=feature.shape[-1], num_layers=2)
        self.model = self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.load(self.load_path)
        self.model.zero_grad()
        seq_length = torch.tensor([feature.shape[1]], dtype=torch.long)
        if torch.cuda.is_available():
            feature = torch.tensor(feature, dtype=torch.float32).cuda()
            seq_length = seq_length.cuda()
        else:
            feature = torch.tensor(feature, dtype=torch.float32)
        _, pred = self.model(feature, seq_length)
        return pred.detach().cpu().numpy()

    def evaluate(self, repo):
        batch_generator = BatchGenerator(repo=repo, mode='test')
        feat_dim = batch_generator.get_feat_dim()
        self.model = Model(num_classes=len(repo.tags()), feat_dim=feat_dim, num_layers=2)
        self.model = self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.load(self.load_path)
        self.model.zero_grad()
        test_data = batch_generator.forward()
        sum, count = 0, 0
        result = np.zeros((9, 9))
        for feature, seq_length, pred in test_data:
            #pdb.set_trace()
            sum += 1
            feature = torch.tensor(feature, dtype=torch.float32)
            seq_length = torch.tensor(seq_length, dtype=torch.long)
            if torch.cuda.is_available():
                feature = feature.cuda()
                seq_length = seq_length.cuda()
            _, pred_y = self.model(feature, seq_length)
            result[pred[0]][int(torch.argmax(pred_y))] += 1
            if int(torch.argmax(pred_y)) == pred[0]:
                count += 1
            torch.cuda.empty_cache()
        print("accuracy: {:.2f}%".format(count / sum * 100))
        #return count / sum
        return result

    def serialize(self, file_name):
        self.model.save(file_name)

    def populate(self, file_name):
        self.load_path = file_name


def msgpack2repo(file_path):
    msg = umsgpack.unpackb(open(file_path, 'rb').read())

    repo = Repository()
    for program in msg['programs']:
        if len(program['funcs']) == 0:
            continue
        name = program['name']
        label = program['label']
        funcs = program['funcs']
        entries = program['entries']

        functions = {}
        for func in msg['funcs'].values():
            func_id = func['id']
            name = func['name']
            v = np.array(func['vec'])

            functions[func_id] = Function(func_id, name, v)

        for func in msg['funcs'].values():
            func_id = func['id']
            callees = func['callees']

        for callee in callees:
            if functions[callee] in funcs:
                functions[func_id].add_callee(functions[callee])

        p = Program(name, ProgramTag(label))
        for entry in entries:
            p.add_entry(functions[entry])
        if len(p.entries()) == 0:
            for func in funcs:
                p.add_entry(functions[func])

        repo.add_program(p)
    
    return repo

    
class BatchGenerator(object):

    def __init__(self, repo, batch_size=32, mode='train'):
        # 所有的特征要pad到相同的长度, 并且按长度降序排列
        super(BatchGenerator, self).__init__()
        if mode == 'train':
            self.batch_size = batch_size
        else:
            self.batch_size = 1
        self.mode = mode
        self.programs = repo.programs()
        t = dict(enumerate(repo.tags()))
        self.tags = {t[i]: i for i in t.keys()}

    def get_feat_dim(self):
        return self.programs[0].entries()[0].vec().shape[-1]

    def forward(self):
        if self.mode == 'train':
            programs = self.programs[:int(len(self.programs)*0.8)]
            while 1:
                np.random.shuffle(programs)
                for i in range(len(programs)//self.batch_size):
                    feature = []
                    seq_length = []
                    pred = []
                    for j in range(i, i+self.batch_size):
                        feature.append(transform(programs[j]))
                        seq_length.append(feature[-1].shape[0])
                        pred.append(self.tags[programs[j].tag()])
                    seq_length = np.array(seq_length)
                    pred = np.array(pred)
                    idx = np.argsort(seq_length)[::-1]
                    seq_length = seq_length[idx]
                    pred = pred[idx]
                    for j in range(0, self.batch_size):
                        feature[j] = np.pad(feature[j], ((0, feature[idx[0]].shape[0]-feature[j].shape[0]), (0, 0)), mode='constant')
                    feature = np.array(feature)
                    feature = feature[idx]
                    yield feature, seq_length, pred
        else:
            programs = self.programs[int(len(self.programs)*0.8):]
            #programs = self.programs
            np.random.shuffle(programs)
            for i in range(len(programs) // self.batch_size):
                feature = []
                seq_length = []
                pred = []
                for j in range(i, i + self.batch_size):
                    feature.append(transform(programs[j]))
                    seq_length.append(feature[-1].shape[0])
                    pred.append(self.tags[programs[j].tag()])
                seq_length = np.array(seq_length)
                pred = np.array(pred)
                idx = np.argsort(seq_length)[::-1]
                seq_length = seq_length[idx]
                pred = pred[idx]
                for j in range(0, self.batch_size):
                    feature[j] = np.pad(feature[j], ((0, feature[0].shape[0] - feature[j].shape[0]), (0, 0)), mode='constant')
                feature = np.array(feature)
                feature = feature[idx]
                yield feature, seq_length, pred
                '''
                    x = np.random.rand(self.batch_size, 50, 200)
                    y = np.random.randint(1, 51 , self.batch_size)
                    for i in range(self.batch_size):
                        x[i][y[i]:] = 0.
                    z = np.random.randint(0, 14, self.batch_size)
                    idx = np.argsort(y)[::-1]
                    x = x[idx]
                    y = y[idx]
                    z = z[idx]
                    yield x, y, z
                '''


class Model(nn.Module):

    def __init__(self, batch_size=32, num_classes=14, feat_dim=400, lstm_size=300, num_layers=2, keep_prob=0.5):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.keep_prob = keep_prob
        self.lstm_size = lstm_size
        self.batch_size = batch_size
        self.feat_dim = feat_dim
        self.rnn = nn.LSTM(self.feat_dim, self.lstm_size, self.num_layers)
        #self.hidden_state = torch.randn(num_layers, batch_size, lstm_size)
        #self.cell_state = torch.randn(num_layers, batch_size, lstm_size)
        temp_size = int(math.sqrt(lstm_size/num_classes))
        self.bn1 = nn.BatchNorm1d(lstm_size)
        self.bn2 = nn.BatchNorm1d(temp_size)
        self.fc1 = nn.Linear(lstm_size, temp_size)
        self.fc2 = nn.Linear(temp_size, temp_size)
        self.fc3 = nn.Linear(temp_size, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(self.keep_prob)

    def forward(self, inputs, seq_length):
        packed_inputs = nn.utils.rnn.pack_padded_sequence(inputs, seq_length, batch_first=True)
        #packed_outputs, (self.hidden_state, self.cell_state) = self.rnn(packed_inputs)
        packed_outputs, _ = self.rnn(packed_inputs)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        idx = torch.tensor(range(seq_length.size()[0]), dtype=torch.long)
        if torch.cuda.is_available():
            idx = idx.cuda()
        idx = idx.reshape(1, -1)
        idx = torch.cat((idx, seq_length.reshape(1, -1)-1), 0)
        #pdb.set_trace()
        outputs = outputs[idx.chunk(2)]
        outputs = outputs.reshape(-1, outputs.shape[-1])
        logits = self.bn1(outputs)
        logits = self.relu(self.fc1(logits))
        logits = self.bn2(self.relu(self.fc2(logits)))
        logits = self.dropout(logits)
        logits = self.relu(self.fc3(logits))
        pred = nn.Softmax(-1)(logits)
        return logits, pred

    def loss(self, logits, pred):
        loss = nn.CrossEntropyLoss(reduction='mean')(logits, pred)

        return loss

    def load(self, path):
        model_dict = self.state_dict()
        pretrained_dict = torch.load(path)
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def save(self, path):
        f = open(path, 'wb')
        torch.save(self.state_dict(), f)
        f.close()


if __name__ == "__main__":
    repo = msgpack2repo("models/vulcls-repo.msgpack")
    model = LSTMModel()
    model.train(repo, batch_size=32, save_path='weights/lstm_model.ckpt', iters=5000, save_per_iter=1000)
    '''repo = msgpack2repo("models/vulcls-repo.msgpack")
    model = LSTMModel('weights/lstm_model_4999.ckpt')
    #prog = repo.programs()[0]
    #print(model.predict(repo, prog))
    result = model.evaluate(repo)
    result = pandas.DataFrame(result)
    result.to_csv("123.csv", index=None)'''