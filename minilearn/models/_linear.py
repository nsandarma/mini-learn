from minilearn.models._base import BaseClassifier,BaseRegressor
from minilearn.utils import sigmoid,softmax
from minilearn.encoders import onehot_encoder
import numpy as np

__all__ = [
  "LinearRegression",
  "LogisticRegression"
]

class LinearRegression(BaseRegressor):
  def __init__(self,fit_intercept=True):
    self.fit_intercept = fit_intercept

  def __add_intercept(self,X):
    intercept = np.ones((X.shape[0],1))
    return np.concatenate((intercept,X),axis=1)
  
  def fit(self,X,y):
    if self.fit_intercept:
      X = self.__add_intercept(X)
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    self.intercept_ = theta[0] if self.fit_intercept else 0.0
    self.__theta = theta[1:] if self.fit_intercept else theta
    return self

  def predict(self,X):
    return X.dot(self.__theta) + self.intercept_

  @property
  def coef_(self)->np.ndarray:
    return self.__theta


class LogisticRegression(BaseClassifier):
  # solver : liblinear
  def __init__(self,lr=0.001,n_iters=100,fit_intercept=True):
    self.lr = lr
    self.n_iters = n_iters
    self.fit_intercept = fit_intercept
    self.linear = lambda x : x.dot(self.coef_.T) + self.intercept_
    
  def fit(self,X,y):
    n_samples,n_features = X.shape
    self.y = y
    y_size = len(np.unique(self.y))
    self.multiclass = y_size > 2
    coef_size = (n_features,) if not self.multiclass else (y_size,n_features)
    intercept_size = (1,) if not self.multiclass else (y_size,)
    self.coef_ = np.zeros(shape=coef_size)
    self.intercept_ = np.zeros(shape=intercept_size) if self.fit_intercept else np.zeros(intercept_size)
    y = onehot_encoder(y.reshape(-1,1)) if y_size > 2 else y

    for _ in range(self.n_iters):
      lin = self.linear(x=X)
      
      pred = softmax(lin) if y_size > 2 else sigmoid(lin)
      
      loss = pred - y
      
      dw = (1/n_samples) * np.dot(loss.T, X)
      db = (1/n_samples) * np.sum(loss,axis=0)
      
      self.coef_ -= self.lr * dw if y_size > 2 else self.lr * dw.flatten()
      self.intercept_ -= self.lr * db
    return self
    
  def predict_proba(self,X): 
    X = self.linear(X)
    if self.multiclass:
      return softmax(X)
    pred = sigmoid(X)
    res = np.array([(1-y) for y in pred])
    return np.column_stack((res,pred))

  def predict(self,X): return self.predict_proba(X).argmax(1)

