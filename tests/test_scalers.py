from sklearn.preprocessing import StandardScaler as SS,MinMaxScaler as MM
from minilearn.scalers import StandardScaler,MinMaxScaler
import unittest
import numpy as np


class TestScaler(unittest.TestCase):
  def setUp(self):
    self.X = np.random.randint(3,1000,size=(100,5))
  
  def test_standard(self):
    x1 = StandardScaler().fit_transform(self.X)
    x2 = SS().fit_transform(self.X)
    np.testing.assert_array_equal(x1,x2)

  def test_minmax(self):
    x1 = MinMaxScaler().fit_transform(self.X)
    x2 = MM().fit_transform(self.X)
    np.testing.assert_array_equal(x1,x2)
  
  def test_inverse_standard(self):
    s1 = StandardScaler().fit(self.X)
    s2 = SS().fit(self.X)
    np.testing.assert_array_equal(s1.inverse_transform(s1.transform(self.X)),s2.inverse_transform(s2.transform(self.X)))

  def test_inverse_minmax(self):
    s1 = MinMaxScaler().fit(self.X)
    s2 = MM().fit(self.X)
    np.testing.assert_array_equal(s1.inverse_transform(s1.transform(self.X)),s2.inverse_transform(s2.transform(self.X)))
  

  
if __name__ == "__main__":
  unittest.main()