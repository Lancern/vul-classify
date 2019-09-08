import numpy as np
import xgboost as xgb

from ..asm import collect_functions

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
            funcs = collect_functions(program.entry())
            vecs = [func.vec() for func in funcs]

        X_train.append(_all_reduce(vecs))
        y_train.append(program.tag())
                
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        self._model.fit(X_train, y_train,
                        eval_set=[(X_train, y_train)],
                        eval_metric='mlogloss')
            
    def predict(self, repo, target):
        funcs = collect_functions(target.entry())
        vecs = [func.vecs() for func in funcs]
        
        X_test = np.array([_all_reduce(vecs)])
        
        y_pred = self._model.predict(X_test)
        pred = [1 if y_pred[0] == tag else 0 for tag in self._tags]
        
        return pred

