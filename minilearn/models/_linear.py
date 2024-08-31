from minilearn.models._base import BaseClassifier,BaseRegressor
from minilearn.models.__utils import sigmoid
import numpy as np


class LogisticRegression(BaseClassifier):
  def __init__(self,learning_rate=0.01,max_iter=1000,fit_intercept=True,verbose=True):
    self.__learning_rate = learning_rate
    self.__max_iter = max_iter
    self.__fit_intercept = fit_intercept
    self.verbose = verbose
  
  def add_intercept(self,X):
    intercept = np.ones((X.shape[0],1))
    return np.concatenate((intercept,X),axis=1)
  
  def loss(self,h,y): return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
  
  def fit(self,X,y):
    if self.__fit_intercept:
      X = self.add_intercept(X)
    
    self.theta  = np.zeros(X.shape[1])

    for i in range(self.__max_iter):
      z = np.dot(X,self.theta)
      h = sigmoid(z)
      gradient = np.dot(X.T,(h-y)) / y.size
      self.theta -= self.__learning_rate * gradient

      if self.verbose and i % 100 == 0:
        print(f"loss : {self.loss(h,y)} \t")
      
  def predict_proba(self,X):
    if self.__fit_intercept:
      X = self.add_intercept(X)
    
    return sigmoid(np.dot(X,self.theta))
  
  def predict(self,X,threshold = 0.5):
    return self.predict_proba(X) >= threshold




class LinearRegression(BaseRegressor):
  def __init__(self):
    pass

if __name__ == "__main__":
  from minilearn.datasets import load_clf,make_clf
  from sklearn.linear_model import LogisticRegression as LG
  x_train,y_train,x_test,y_test = make_clf(n_features=5,n_samples=100,train_size=0.2,norm=False)

  model = LogisticRegression(verbose=True)
  model.fit(x_train,y_train)

  # print(model.score(x_test,y_test))
  # print(model.predict(x_test))
