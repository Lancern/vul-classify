import math

import numpy as np

import torch
import torch.nn as nn
from torch import optim

from .base import AbstractModel


def transform(prog, precesion=.01):
    functions = {}
    queue = [prog.entry()]
    functions[prog.entry().id()] = prog.entry()
    start, end = 0, 1
    while start < end:
        start_func = queue[start]
        start += 1
        for func in start_func.callees():
            if func.id() not in functions:
                queue.append(func)
                end += 1
                functions[func.id()] = func
    num = len(functions.keys())
    weight_feature = np.array([1. for _ in range(num)])
    for _ in range(10):
        count = 0
        for i in range(num):
            temp = 0
            for func in functions[i].callers():
                temp += weight_feature[func.id()]/len(func.callees())
            count += temp-weight_feature[i]
            weight_feature[i] = temp
        if count/num < precesion:
            break
    idx = np.argsort(weight_feature)[::-1]
    feature = np.array([functions[i].vec() for i in idx])
    return feature.reshape(feature[0], -1)


class LSTMModel(AbstractModel):

    def __init__(self):
        pass

    def train(self, repo, batch_size=32, load_path=None, save_path=None, save_per_iter=5000, iters=10000, lr=0.001, lstm_size=200, num_layers=2, keep_prob=.5):
        batch_generator = BatchGenerator(repo, batch_size)
        model = Model(batch_size, len(repo.tags()), feat_dim, lstm_size, num_layers, keep_prob)
        model = model.train()
        if torch.cuda.is_available():
            model = model.cuda()
        model.zero_grad()
        if load_path is not None:
            model.load(load_path)
        optimizer = optim.SGD(model.parameters(), lr)
        train_data = batch_generator.forward()
        for i in range(iters):
            optimizer.zero_grad()
            feature, seq_length, pred = next(train_data)
            if not torch.cuda.is_available():
                feature = torch.tensor(feature, dtype=torch.float32)
                seq_length = torch.tensor(seq_length, dtype=torch.long)
                pred = torch.tensor(pred, dtype=torch.long)
            else:
                feature = feature.cuda()
                seq_length = seq_length.cuda()
                pred = pred.cuda()
            logits, _ = model.forward(feature, seq_length)
            loss = model.loss(logits, pred)
            loss.backward()
            optimizer.step()
            if (i + 1) % save_per_iter == 0:
                model.load(save_path)
            torch.cuda.empty_cache()

    def predict(self, repo, prog):
        feature = transform(prog)
        assert len(feature.shape) == 3 and feature.shape[0] == 1, "feature must be [1, seq_length, feat_dim]"
        model = Model(feat_dim=feature.shape[-1])
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        model.load(load_path)
        model.zero_grad()
        seq_length = torch.tensor([feature.shape[1]], dtype=torch.Long)
        if torch.cuda.is_available():
            feature = torch.tensor(feature, dtype=torch.float32).cuda()
            seq_length = seq_length.cuda()
        else:
            feature = torch.tensor(feature, dtype=torch.float32)
        _, pred = model.forward(feature, seq_length)
        return pred.detach().numpy()

    def evaluate(self, repo, load_path):
        batch_generator = BatchGenerator(repo=repo, mode='test')
        model = Model()
        model = model.eval()
        if torch.cuda.is_avaliable():
            model = model.cuda()
        model.load(load_path)
        model.zero_grad()
        test_data = batch_generator.forward()
        sum, count = 0, 0
        for feature, seq_length, pred in test_data:
            sum += 1
            if not torch.cuda.is_available():
                feature = torch.tensor(feature, dtype=torch.float32)
                seq_length = torch.tensor(seq_length, dtype=torch.long)
            else:
                feature = feature.cuda()
                seq_length = seq_length.cuda()
            _, pred_y = model.forward(feature, seq_length)
            if int(torch.argmax(pred_y)) == pred:
                count += 1
            torch.cuda.empty_cache()
        print("accuracy: {:.2f}%".format(count / sum * 100))
        return count / sum


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
        t = enumerate(repo.tags())
        self.tags = {t[i]: i for i in t.keys()}

    def forward(self):
        if mode == 'train':
            while 1:
                np.random.shuffle(self.programs)
                for i in range(len(self.programs)//self.batch_size):
                    feature = []
                    seq_length = []
                    pred = []
                    for j in range(i, i+self.batch_size):
                        feature.append(transform(self.programs[j]))
                        seq_length.append(feature[-1].shape[0])
                        pred.append(self.tags[self.programs[j].tag()])
                    feature = np.array(feature)
                    seq_length = np.array(seq_length)
                    pred = np.array(pred)
                    idx = np.argsort(seq_length)[::-1]
                    feature = feature[idx]
                    seq_length = seq_length[idx]
                    pred = pred[idx]
                    for j in range(1, self.batch_size):
                        feature[j] = np.pad(feature[j], (0, feature[0].shape[0]-[j].shape[0]), mode='constant')
                    yield feature, seq_length, pred
        else:
            for i in range(len(self.programs) // self.batch_size):
                feature = []
                seq_length = []
                pred = []
                for j in range(i, i + self.batch_size):
                    feature.append(transform(self.programs[j]))
                    seq_length.append(feature[-1].shape[0])
                    pred.append(self.tags[self.programs[j].tag()])
                feature = np.array(feature)
                seq_length = np.array(seq_length)
                pred = np.array(pred)
                idx = np.argsort(seq_length)[::-1]
                feature = feature[idx]
                seq_length = seq_length[idx]
                pred = pred[idx]
                for j in range(1, self.batch_size):
                    feature[j] = np.pad(feature[j], (0, feature[0].shape[0] - [j].shape[0]), mode='constant')
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

    def __init__(self, batch_size=1, num_classes=14, feat_dim=200, lstm_size=200, num_layers=2, keep_prob=0.5):
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
        self.fc1 = nn.Linear(lstm_size, temp_size)
        self.fc2 = nn.Linear(temp_size, num_classes)
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
        outputs = outputs[idx.chunk(2)]
        outputs = outputs.reshape(-1, outputs.shape[-1])
        logits = self.relu(self.fc1(outputs))
        logits = self.dropout(logits)
        logits = self.relu(self.fc2(logits))
        pred = nn.Softmax(-1)(logits)
        return logits, pred

    def loss(self, logits, pred):
        loss = nn.CrossEntropyLoss(reduction='mean')(logits, pred)

        return loss

    def load(self, path):
        model_dict = self.state_dict()
        pretrained_dict = torch.load(path)
        model_dict.update(pretrained_dict)

    def save(self, path):
        f = open(path, 'wb')
        torch.save(self.state_dict(), f)
        f.close()
