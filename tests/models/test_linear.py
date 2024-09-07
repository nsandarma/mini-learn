import unittest 

from minilearn.models import LinearRegression,LogisticRegression
from minilearn.datasets import load_reg,load_clf
from sklearn import linear_model
import numpy as np


class TestLinearRegression(unittest.TestCase):
  # def setUp(self) -> None:
  #   self.x_train,self.y_train,self.x_test,self.y_test = load_reg()
  
  def test_predict(self):
    n_iters = 10
    for _ in range(n_iters):
      x_train,y_train,x_test,_ = load_reg()
      model = LinearRegression().fit(x_train,y_train)
      pred1 = model.predict(x_test)
      model  = linear_model.LinearRegression().fit(x_train,y_train)
      pred2 = model.predict(x_test)
      np.testing.assert_array_equal(pred1.round(5),pred2.round(5))

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
    