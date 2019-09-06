import os
import xgboost as xgb
import numpy as np

import vul_classify.repr
from models import AbstractModel


def all_reduce(vecs):
    n_vecs = len(vecs)
    n_dimensions = len(vecs[0])
    
    X = [0 for i in range(n_dimensions)]
    for vec in vecs:
        X = [a + b for a, b in zip(X, vec)]
        X = [x / n_vecs for x in X]
    
    return X
    
class XGBModel(AbstractModel):
    def __init__(self, **kwargs):
        self.model = xgb.XGBClassifier(**kwargs)
        
    def train(self, repo):
        self.tags = repo.tags()
        
        programs = repo.programs()
        
        X_train = []
        y_train = []
        for program in programs:
            funcs = vul_classify.repr.collect_functions(program.entry())
            vecs = [func.vec() for func in funcs]
            
        X_train.append(all_reduce(vecs))
        y_train.append(program.tag())
                
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        self.model.fit(X_train, y_train,
                    eval_set=[(X_train, y_train)],
                        eval_metric='mlogloss')
            
    def predict(self, target):
        funcs = vul_classify.repr.collect_functions(target.entry())
        vecs = [func.vecs() for func in funcs]
        
        X_test = np.array([all_reduce(vecs)])
        
        y_pred = self.model.predict(X_test)
        pred = [1 if y_pred[0] == tag else 0 for tag in self.tags]
        
        return pred
        
