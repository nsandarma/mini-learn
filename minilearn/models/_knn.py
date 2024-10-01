import numpy as np
from typing import Literal
from collections import Counter

from minilearn.models._base import BaseClassifier,BaseRegressor
from minilearn.utils import minhowski_distance

__all__ = [
  "KNN",
  "KNNRegressor",
  "MKNN"
]

class NearestNeighbor:
  def __init__(self,n_neighbors:int = 5 , metric:Literal["euclidean","manhattan"] ="euclidean") :
    assert metric in ["euclidean","manhattan"], "metric not in 'euclidean','manhattan']"
    self.__n_neighbors = n_neighbors
    self.__metric = metric

  def fit(self,X:np.ndarray,y:np.ndarray):
    assert len(X) == len(y) , "len is not same !"
    self.X = X
    self.n_samples_,self.n_features_in_ = X.shape
    if BaseClassifier in self.__class__.__bases__:
      self.classes_,self.class_count_ = np.unique(y,return_counts=True)
    self.y = y
    self.n = len(X)
    self.is_fitted = True
    return self

  def neighbors(self,X:np.ndarray = None,n_neighbors:int = None) -> tuple:
    """return : dist,indices"""
    self.check_is_fitted()

    x_test = self.X if X is None else X
    n_neighbors = self.n_neighbors if n_neighbors is None else n_neighbors
    x_test = np.array(x_test) if not isinstance(x_test,np.ndarray) else x_test

    assert x_test.ndim == 2 and x_test.shape[1] == self.X.shape[1], "X_test dim not same with X_train"
    p = 1 if self.metric == "manhattan" else 2
    distances = np.array([minhowski_distance(self.X,x,p=p) for x in x_test])

    if x_test is self.X and X is None: np.fill_diagonal(distances, np.inf)
    args = np.argsort(distances,1)[:,:n_neighbors]
    distances = np.take_along_axis(distances, args, axis=1)
    
    return distances, args
  
  @property
  def n_neighbors(self)->int:return self.__n_neighbors

  @property
  def metric(self)->str: return self.__metric


class KNN(NearestNeighbor,BaseClassifier):
  name = "KNN"
  t = "classification"

  def predict(self,X:np.ndarray):
    _,idx = self.neighbors(n_neighbors=self.n_neighbors,X = X)
    __voting = lambda idx : Counter(self.y[idx]).most_common(1)[0][0] if len(idx) != 2 else self.y[idx][1]
    y_pred = np.array([__voting(i) for i in idx])
    return y_pred

  def predict_proba(self, X: np.ndarray):
      _, idx = self.neighbors(X = X)
      probabilities = []
      for neighbors_indices in idx:
          class_counts = Counter(self.y[neighbors_indices])
          total_counts = sum(class_counts.values())
          proba = {cls: count / total_counts for cls, count in class_counts.items()}
          probabilities.append([proba.get(cls, 0.0) for cls in np.unique(self.y)])
      return np.array(probabilities)

class MKNN(KNN):
  name = "MKNN"

  def __init__(self,n_neighbors:int = 5,metric:Literal["euclidean","manhattan"] = "euclidean",e:float =0.5):
    self.e = e
    super().__init__(n_neighbors=n_neighbors,metric=metric)

  def __weight_calculate(self,dist,idx,val):
    weight = val * 1 / (dist+self.e) 
    argm = weight.argmax()
    return self.y[idx[argm]]
    
  def predict(self,X):
    validity = self.validity
    dist,idx = self.neighbors(X = X)
    return np.array([self.__weight_calculate(dist[i],idx[i],validity[idx[i]]) for i in range(len(X))])
  
  @property
  def validity(self):
    _,idx = self.neighbors()
    __voting = lambda idx : Counter(self.y[idx]).most_common(1)[0][0]
    _y = np.array([__voting(i) for i in idx])
    return np.array([1 if _y[i] == self.y[i] else 0 for i in range(self.n)])

class KNNRegressor(NearestNeighbor,BaseRegressor):
  name = "KNNRegressor"
  t = "regression"
  
  def __predict(self,idx): return np.mean(self.y[idx])

  def predict(self,X:np.ndarray):
    _,idx = self.neighbors(X)
    y_pred = np.array([self.__predict(i) for i in idx])
    return y_pred
  
