import unittest
from minilearn.encoders import OneHotEncoder,LabelEncoder,OrdinalEncoder,TargetEncoder,LabelBinarizer
from minilearn.metrics import MAE
import pandas as pd
from sklearn import preprocessing
import numpy as np
import time

df = pd.read_csv("examples/dataset/drug200.csv")
cats = df.select_dtypes("object").values

class TestOneHotEncoder(unittest.TestCase):
  def setUp(self):
    self.x = cats
    self.c = cats[:100]
  
  def test_transform(self):
    enc = OneHotEncoder()
    enc.fit(self.x)
    x_transform1 =  enc.transform(self.c)

    enc = preprocessing.OneHotEncoder()
    enc.fit(self.x)
    x_transform2 = enc.transform(self.c).toarray()

    np.testing.assert_array_equal(x_transform1,x_transform2)
  
  def test_inverse_transform(self):
    enc = OneHotEncoder()
    enc.fit(self.x)
    x_transform1 = enc.transform(self.c)
    inverse_transform1 = enc.inverse_transform(x_transform1)

    enc = preprocessing.OneHotEncoder()
    enc.fit(self.x)
    x_transform2 = enc.transform(self.c).toarray()
    inverse_transform2 = enc.inverse_transform(x_transform2)

    np.testing.assert_array_equal(inverse_transform1,inverse_transform2)
  
  def test_performance(self):
    start = time.monotonic()
    enc = OneHotEncoder().fit(self.x)
    x_transform = enc.transform(self.c)
    enc.inverse_transform(x_transform)
    times = time.monotonic() - start

    start = time.monotonic()
    enc = preprocessing.OneHotEncoder().fit(self.x)
    x_transform = enc.transform(self.c)
    enc.inverse_transform(x_transform)
    time_threshold = time.monotonic() - start
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
  

class TestOrdinalEncoder(unittest.TestCase):
  def setUp(self):
    self.x = cats
    self.c = cats[np.random.randint(0,len(cats),(100,))]

  def test_transform(self):
    enc = OrdinalEncoder()
    enc.fit(self.x)
    x_transform1 = enc.transform(self.c)

    enc = preprocessing.OrdinalEncoder()
    enc.fit(self.x)
    x_transform2 = enc.transform(self.c)
    
    np.testing.assert_array_equal(x_transform1,x_transform2)

  def test_inverse_transform(self):

    enc = OrdinalEncoder()
    enc.fit(self.x)
    inverse_transform1 = enc.inverse_transform(enc.transform(self.c))

    enc = preprocessing.OrdinalEncoder()
    enc.fit(self.x)
    inverse_transform2 = enc.inverse_transform(enc.transform(self.c))
    
    np.testing.assert_array_equal(inverse_transform1,inverse_transform2)
  
  def test_performance(self):
    start = time.monotonic()
    enc = OrdinalEncoder().fit(self.x)
    x_transform = enc.transform(self.c)
    enc.inverse_transform(x_transform)
    times = time.monotonic() - start

    start = time.monotonic()
    enc = preprocessing.OrdinalEncoder().fit(self.x)
    x_transform = enc.transform(self.c)
    enc.inverse_transform(x_transform)
    time_threshold = time.monotonic() - start

    self.assertLess(times,time_threshold)

  def test_checkisin(self):
    # test pesan error ketika x transform tidak ada di x fit (isNotMember)
    x = np.array(["a","b","c"]).reshape(-1,1)
    s = np.array(["a","c","d"]).reshape(-1,1)
    enc = OrdinalEncoder()
    enc.fit(x)
    with self.assertRaises(Exception) as context:
      enc.transform(s)
    err_msg = "Found unknown categories ['d'] in column 0 during transform"
    self.assertEqual(str(context.exception),err_msg)
  
  def test_n_features(self):
    # test pesan error ketika n_features pada transform berbeda dengan di fit (shape difference)
    x = np.array([["a","b","c"],["a","b","c"]])
    s = np.array(["a","c","d"]).reshape(-1,1)
    enc = OrdinalEncoder()
    enc.fit(x)
    with self.assertRaises(Exception) as context:
      enc.transform(s)
    err_msg = "X has 1 features, but OneHotEncoder is expecting 3 features as input."
    self.assertEqual(str(context.exception),err_msg)


class TestTargetEncoder(unittest.TestCase):
  def setUp(self):
    features = df.select_dtypes("object").columns
    self.x = df[features].values
    self.y = df["Drug"].values
  
  def test_attribute(self):
    smooth = 2
    enc = TargetEncoder(smooth=2).fit(self.x,self.y)
    enc2 = preprocessing.TargetEncoder(smooth=smooth).fit(self.x,self.y)

    #encodings_
    for i,c in enumerate(enc.encodings_):
      np.testing.assert_allclose(c,enc2.encodings_[i])
    
    #categories_
    for i,cat in enumerate(enc.categories_):
      np.testing.assert_array_equal(cat,enc2.categories_[i])

    np.testing.assert_array_equal(enc.classes_,enc2.classes_) 
    

  def test_transform(self):
    smooth = 2
    enc = TargetEncoder(smooth=smooth).fit(self.x,self.y)
    x_transform1 = enc.transform(self.x)
    enc = preprocessing.TargetEncoder(smooth=smooth).fit(self.x,self.y)
    x_transform2 = enc.transform(self.x)

    # there may be differences here, but on a very small scale
    np.testing.assert_allclose(x_transform1,x_transform2)
  
  def test_performance(self):
    smooth = 2

    start = time.monotonic()
    enc = TargetEncoder(smooth=smooth).fit(self.x,self.y)
    x_transform = enc.transform(self.x)
    times = time.monotonic() - start

    start = time.monotonic()
    enc = preprocessing.TargetEncoder(smooth=smooth).fit(self.x,self.y)
    x_transform = enc.transform(self.x)
    time_threshold = time.monotonic() - start
    self.assertLess(abs(times - time_threshold),0.01)


class TestLabelEncoder(unittest.TestCase):
  def setUp(self):
    self.y = df["Drug"].values
    self.c = self.y[np.random.randint(0,len(cats),(100,))]

  def test_transform(self):
    enc = LabelEncoder().fit(self.y)
    y_transform = enc.transform(self.c)

    enc = preprocessing.LabelEncoder().fit(self.y)
    y_transform1 = enc.transform(self.c)

    np.testing.assert_array_equal(y_transform,y_transform1)
    np.testing.assert_array_equal(self.c,enc.inverse_transform(y_transform))
  
  def test_performance(self):
    start = time.monotonic()
    enc = LabelEncoder().fit(self.y)
    y_transform = enc.transform(self.c)
    enc.inverse_transform(y_transform)
    times = time.monotonic() - start


    start = time.monotonic()
    enc = preprocessing.LabelEncoder().fit(self.y)
    y_transform1 = enc.transform(self.c)
    enc.inverse_transform(y_transform1)
    time_threshold = time.monotonic() - start

    self.assertLess(times,time_threshold)

class TestLabelBinarizer(unittest.TestCase):
  def setUp(self):
    self.y = df["Drug"].values
  
  def test_transform(self):
    enc = LabelBinarizer().fit(self.y)
    y_transform = enc.transform(self.y)

    enc = preprocessing.LabelBinarizer().fit(self.y)
    y_transform1 = enc.transform(self.y)

    np.testing.assert_array_equal(y_transform,y_transform1)
  
  def test_inverse_transform(self):
    enc = LabelBinarizer().fit(self.y)
    y_transform = enc.transform(self.y)
    np.testing.assert_array_equal(enc.inverse_transform(y_transform),self.y)

  
if __name__ == "__main__":
  unittest.main()

