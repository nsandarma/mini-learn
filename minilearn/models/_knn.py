import numpy as np
from typing import Literal
from collections import Counter

from minilearn.models._base import Base
from minilearn.models.__utils import minhowski_distance


class KNN(Base):
  name = "KNN"
  t = "classification"

  def __init__(self,n_neighbors:int = 5,metric:Literal["euclidean","manhattan"]="euclidean"):
    assert metric in ["euclidean","manhattan"], "metric not in 'euclidean','manhattan']"
    self.n_neighbors = n_neighbors
    self.metric = metric

  def fit(self,X_train:np.ndarray,y_train:np.ndarray):
    assert len(X_train) == len(y_train) , "len is not same !"
    self.X_train = X_train
    self.y_train = y_train
    self.n = len(X_train)
    self.is_fitted = True
    return self
  
  def fit_predict(self,X_train:np.ndarray,y_train:np.ndarray):
    self.fit(X_train,y_train)

  def _predict(self,idx:np.ndarray):
    return Counter(self.y_train[idx]).most_common(1)[0][0]

  def predict(self,x_test:np.ndarray):
    self.check_is_fitted()
    _,idx = self.neighbors(x_test)
    y_pred = np.array([self._predict(i) for i in idx])
    return y_pred

  def neighbors(self,x_test:np.ndarray) -> tuple:
    """return : dist,indices"""
    self.check_is_fitted()
    x_test = np.array(x_test) if not isinstance(x_test,np.ndarray) else x_test
    assert x_test.ndim == 2 and x_test.shape[1] == self.X_train.shape[1], "X_test dim not same with X_train"

    p = 1 if self.metric == "manhattan" else 2

    distances = np.array([minhowski_distance(self.X_train,x,p=p) for x in x_test])

    args = np.argsort(distances,1)[:,:self.n_neighbors]
    distances.sort(axis=1)
    return distances[:,:self.n_neighbors],args

class KNNRegressor(KNN):
  name = "KNNRegressor"
  t = "regression"
  def __init__(self,n_neighbors:int = 5,metric = Literal["euclidean","manhattan"]):
    super().__init__(n_neighbors=n_neighbors,metric=metric)
  
  def _predict(self,idx):
    return np.mean(self.y_train[idx])
  

class MKNN(KNN):
  name = "MKNN"
  def __init__(self,n_neighbors:int = 5,metric:Literal["euclidean","manhattan"] = "euclidean",e:float =0.5):
    self.e = e
    super().__init__(n_neighbors=n_neighbors,metric=metric)

  def _weight_calculate(self,dist,idx,val):
    weight = val * 1 / (dist+self.e) 
    argm = weight.argmax()
    return self.y_train[idx[argm]]
    
  def predict(self,x_test):
    self.check_is_fitted()
    validity = self.validity
    dist,idx = self.neighbors(x_test)
    return np.array([self._weight_calculate(dist[i],idx[i],validity[idx[i]]) for i in range(len(x_test))])
  
  @property
  def validity(self):
    self.check_is_fitted()
    _y = super().predict(self.X_train)
    return np.array([1 if _y[i] == self.y_train[i] else 0 for i in range(self.n)])


if __name__ == "__main__":
  from sklearn.datasets import make_regression
  from sklearn.neighbors import KNeighborsRegressor
  from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error,r2_score
  np.random.seed(42)

  X,y = make_regression(10000,n_features=100)
  n = int(0.9 * len(X))

  X_train,y_train = X[:n] , y[:n]
  X_test,y_test = X[n:], y[n:]
  
  k = 3

  knn_sk = KNeighborsRegressor(n_neighbors=k,metric="euclidean")
  knn_sk.fit(X_train,y_train)
  print(knn_sk.score(X_test,y_test))

  knn = KNNRegressor(n_neighbors=k,metric="euclidean")
  knn.fit(X_train,y_train)
  print(knn.score(X_test,y_test))








  

  