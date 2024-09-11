import unittest
from minilearn.encoders import OneHotEncoder,LabelEncoder,OrdinalEncoder
import pandas as pd
from sklearn import preprocessing
import numpy as np
import time

df = pd.read_csv("examples/dataset/drug200.csv")
cats = df.select_dtypes("object").values

class TestOneHot(unittest.TestCase):
  def setUp(self):
    self.x = cats
    self.c = cats[:100]
  
  def test_transform(self):
    start = time.perf_counter()
    enc = OneHotEncoder()
    enc.fit(self.x)
    x_transform1 =  enc.transform(self.c)
    times = time.perf_counter() - start

    start = time.perf_counter()
    enc = preprocessing.OneHotEncoder()
    enc.fit(self.x)
    x_transform2 = enc.transform(self.c).toarray()
    time_threshold = time.perf_counter() - start

    np.testing.assert_array_equal(x_transform1,x_transform2)
    self.assertLess(times,time_threshold)
  
  def test_inverse_transform(self):
    start = time.perf_counter()
    enc = OneHotEncoder()
    enc.fit(self.x)
    x_transform1 = enc.transform(self.c)
    inverse_transform1 = enc.inverse_transform(x_transform1)
    times = time.perf_counter() - start

    start = time.perf_counter()
    enc = preprocessing.OneHotEncoder()
    enc.fit(self.x)
    x_transform2 = enc.transform(self.c).toarray()
    inverse_transform2 = enc.inverse_transform(x_transform2)
    time_threshold = time.perf_counter() - start

    np.testing.assert_array_equal(inverse_transform1,inverse_transform2)
    self.assertLess(times,time_threshold)
  
  def test_checkisin(self):
    # test pesan error ketika x transform tidak ada di x fit (isNotMember)
    x = np.array(["a","b","c"]).reshape(-1,1)
    s = np.array(["a","c","d"]).reshape(-1,1)
    enc = OneHotEncoder()
    enc.fit(x)
    with self.assertRaises(Exception) as context:
      enc.transform(s)
    err_msg = "Found unknown categories ['d'] in column 0 during transform"
    self.assertEqual(str(context.exception),err_msg)
  
  def test_n_features(self):
    # test pesan error ketika n_features pada transform berbeda dengan di fit (shape difference)
    x = np.array([["a","b","c"],["a","b","c"]])
    s = np.array(["a","c","d"]).reshape(-1,1)
    enc = OneHotEncoder()
    enc.fit(x)
    with self.assertRaises(Exception) as context:
      enc.transform(s)
    err_msg = "X has 1 features, but OneHotEncoder is expecting 3 features as input."
    self.assertEqual(str(context.exception),err_msg)
  

class TestOrdinal(unittest.TestCase):
  def setUp(self):
    self.x = cats
    self.c = cats[np.random.randint(0,len(cats),(100,))]

  def test_transform(self):
    
    start = time.perf_counter()
    enc = OrdinalEncoder()
    enc.fit(self.x)
    x_transform1 = enc.transform(self.c)
    times = time.perf_counter() - start

    start = time.perf_counter()
    enc = preprocessing.OrdinalEncoder()
    enc.fit(self.x)
    x_transform2 = enc.transform(self.c)
    time_threshold = time.perf_counter() - start
    
    np.testing.assert_array_equal(x_transform1,x_transform2)
    self.assertLess(times,time_threshold)

  def test_inverse_transform(self):

    start = time.perf_counter()
    enc = OrdinalEncoder()
    enc.fit(self.x)
    inverse_transform1 = enc.inverse_transform(enc.transform(self.c))
    times = time.perf_counter() - start

    start = time.perf_counter()
    enc = preprocessing.OrdinalEncoder()
    enc.fit(self.x)
    inverse_transform2 = enc.inverse_transform(enc.transform(self.c))
    time_threshold = time.perf_counter() - start
    
    np.testing.assert_array_equal(inverse_transform1,inverse_transform2)
    self.assertLess(times,time_threshold)

if __name__ == "__main__":
  unittest.main()
