import os
import numpy as np
import xgboost as xgb
import pickle

from .base import AbstractModel


def _all_reduce(vecs):
    n_vecs = len(vecs)
    n_dimensions = len(vecs[0])
    
    X = [0 for _ in range(n_dimensions)]
    for vec in vecs:
        X = [a + b for a, b in zip(X, vec)]
        X = [x / n_vecs for x in X]
    
    return X


class XGBModel(AbstractModel):
    def __init__(self, **kwargs):
        self._model = xgb.XGBClassifier(**kwargs)
        
    def train(self, repo):
        self._tags = repo.tags()
        
        programs = repo.programs()
        
        X_train = []
        y_train = []
        for program in programs:
            funcs = program.funcs()
            if len(funcs) == 0:
                continue

            vecs = [func.vec() for func in funcs]
        
            X_train.append(_all_reduce(vecs))
            y_train.append(program.tag().label())

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        self._model.fit(X_train, y_train,
                        eval_set=[(X_train, y_train)],
                        eval_metric='mlogloss',
                        verbose=False)

    def predict(self, repo, target):
        if repo is not None:
            self.train(repo)

        funcs = target.funcs()
        vecs = [func.vec() for func in funcs]
        
        X_test = np.array([_all_reduce(vecs)])
        
        y_pred = self._model.predict_proba(X_test)
        
        return y_pred

    def serialize(self, file_name):
        # self._model.save_model(file_name)
        with open(file_name, 'wb') as model_file:
            pickle.dump(self._model, model_file)


    def populate(self, file_name):
        # self._model.load_model(file_name)
        with open(file_name, 'rb') as model_file:
            self._model = pickle.load(model_file)

__all__ = ['XGBModel']
