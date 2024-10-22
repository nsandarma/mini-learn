import unittest 

from minilearn.models import LinearRegression,LogisticRegression
from examples.dataset import load_reg,load_clf
from sklearn import linear_model
import numpy as np


class TestLinearRegression(unittest.TestCase):
  
  def test_predict(self):
    n_iters = 10
    for _ in range(n_iters):
      x_train,y_train,x_test,_ = load_reg()
      model = LinearRegression().fit(x_train,y_train)
      pred1 = model.predict(x_test)
      model  = linear_model.LinearRegression().fit(x_train,y_train)
      pred2 = model.predict(x_test)
      np.testing.assert_allclose(pred1,pred2)

  def test_attribute(self):
    x_train,y_train,x_test,y_test = load_reg(n_samples=100,n_features=2)
    model = LinearRegression(fit_intercept=False).fit(x_train,y_train)
    modelr = linear_model.LinearRegression(fit_intercept=False).fit(x_train,y_train)
    np.testing.assert_allclose(model.coef_,modelr.coef_)
    np.testing.assert_allclose(model.intercept_,modelr.intercept_)

    

class TestLogisticRegression(unittest.TestCase):

  def test_predict(self):
    n_iters = 10
    sc1 = []
    sc2 = []
    for _ in range(n_iters):
      x_train,y_train,x_test,y_test = load_clf()
      model = LogisticRegression().fit(x_train,y_train)
      sc1.append(model.score(x_test,y_test))
      model = linear_model.LogisticRegression().fit(x_train,y_train)
      sc2.append(model.score(x_test,y_test))
    assert np.abs(np.mean(sc1) - np.mean(sc2)) <= 0.5 , "err > 0.5"




if __name__ == "__main__":
  unittest.main()
    
