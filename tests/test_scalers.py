from sklearn.preprocessing import StandardScaler as SS,MinMaxScaler as MM
from minilearn.datasets import read_csv
from minilearn.scalers import StandardScaler,MinMaxScaler,minmax,standard
import unittest
import numpy as np


class TestScaler(unittest.TestCase):
  def setUp(self):
    self.X = np.random.randint(3,1000,size=(100,5))
    ds = read_csv("examples/dataset/drug200.csv")
    self.X_ = ds.select_dtypes(exclude="object")

  
  def test_standard(self):
    x1 = StandardScaler().fit_transform(self.X)
    x2 = SS().fit_transform(self.X)
    np.testing.assert_array_equal(x1,x2)

    x1  = StandardScaler().fit_transform(self.X_)
    x2  = SS().fit_transform(self.X_.values)
    np.testing.assert_array_equal(x1,x2)

  def test_minmax(self):
    x1 = MinMaxScaler().fit_transform(self.X)
    x2 = MM().fit_transform(self.X)
    np.testing.assert_array_equal(x1,x2)

    x1  = MinMaxScaler().fit_transform(self.X_)
    x2  = MM().fit_transform(self.X_.values)
    np.testing.assert_array_equal(x1,x2)
  
  def test_inverse_standard(self):
    s1 = StandardScaler().fit(self.X)
    s2 = SS().fit(self.X)
    np.testing.assert_array_equal(s1.inverse_transform(s1.transform(self.X)),s2.inverse_transform(s2.transform(self.X)))

    s1 = StandardScaler().fit(self.X_)
    s2 = SS().fit(self.X_.values)
    np.testing.assert_array_equal(s1.inverse_transform(s1.transform(self.X_)),s2.inverse_transform(s2.transform(self.X_.values)))


  def test_inverse_minmax(self):
    s1 = MinMaxScaler().fit(self.X)
    s2 = MM().fit(self.X)
    np.testing.assert_array_equal(s1.inverse_transform(s1.transform(self.X)),s2.inverse_transform(s2.transform(self.X)))
  
  def test_x_norm(self):
    X = np.random.rand(10,4)
    x1 = MinMaxScaler().fit_transform(X)
    x2 = MM().fit_transform(X)
    np.testing.assert_array_equal(x1,x2)
    x1 = StandardScaler().fit_transform(X)
    x2 = SS().fit_transform(X)
    np.testing.assert_array_equal(x1,x2)
  
  def test_func(self):
    x1 = minmax(self.X)
    x2 = MM().fit_transform(self.X)
    np.testing.assert_array_equal(x1,x2)



  
if __name__ == "__main__":
  unittest.main()
