from minilearn.models._base import BaseClassifier,BaseRegressor
from minilearn.models.__utils import sigmoid,softmax
from minilearn.scalers import is_scaled,minmax
from minilearn.encoders import onehot
import numpy as np


class LinearRegression(BaseRegressor):
  def __init__(self,fit_intercept=True):
    self.__fit_intercept = fit_intercept

  def __add_intercept(self,X):
    intercept = np.ones((X.shape[0],1))
    return np.concatenate((intercept,X),axis=1)
  
  def fit(self,X,y):
    if self.__fit_intercept:
      X = self.__add_intercept(X)
    
    self.__theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return self

  def predict(self,X):
    if self.__fit_intercept:
      X = self.__add_intercept(X)
    return X.dot(self.__theta)

class LogisticRegression:
  def __init__(self,lr=0.001,n_iters=100,fit_intercept=True):
    self.lr = lr
    self.n_iters = n_iters
    self.fit_intercept = fit_intercept
    
  
  def fit(self,X,y):
    y = onehot(y)
    self.w = np.zeros((y.shape[1],X.shape[1]))
    self.b = np.zeros(y.shape[1]) if self.fit_intercept else 0
    for _ in range(self.n_iters):
      pred = self.predict_proba(X)
      err = pred - y
      grad = np.dot(err.T,X)
      self.w -= self.lr * grad
      if self.fit_intercept:
          grad_b = np.sum(err, axis=0)
          self.b -= self.lr * grad_b
    
  def predict_proba(self,X): return softmax(X.dot(self.w.T) + self.b)

  def predict(self,X): self.predict_proba(X).argmax(1)


  




if __name__ == "__main__":
  from minilearn.datasets import load_clf,make_clf,load_reg,make_reg
  from minilearn.metrics import accuracy
  from sklearn import linear_model

  x_train,y_train,x_test,y_test = make_reg(n_samples=1000,n_features=50,train_size=0.7,norm=False)
  model = LinearRegression()
  model.fit(x_train,y_train)
  print(model.score(x_test,y_test))

  model1 = linear_model.LinearRegression()
  model1.fit(x_train,y_train)
  print(model1.score(x_test,y_test))
