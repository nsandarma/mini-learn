import unittest
from minilearn.encoders import OneHotEncoder,LabelEncoder
import pandas as pd
from sklearn import preprocessing
import numpy as np
import time


class TestOneHot(unittest.TestCase):
  def setUp(self):
    df = pd.read_csv("examples/dataset/drug200.csv")
    cats = df.select_dtypes("object").values
    self.x = cats
    self.c = cats
  
  def test_transform(self):
    # for example 
    x = np.array(["a","b","c"]).reshape(-1,1)
    c = np.array(["x","b","c","d"]).reshape(-1,1)
    
    start = time.monotonic()
    enc = OneHotEncoder()
    enc.fit(self.x)
    x_transform1 =  enc.transform(self.c)
    print(x_transform1)
    print(f"times (minilearn): {round(time.monotonic() - start,4)}")

    start = time.monotonic()
    enc = preprocessing.OneHotEncoder()
    enc.fit(self.x)
    x_transform2 = enc.transform(self.c).toarray()
    print(x_transform2)
    print(f"times (sklearn) : {round(time.monotonic() - start,4)}")

    np.testing.assert_array_equal(x_transform1,x_transform2)
  
  def test_inverse_transform(self):
    start = time.monotonic()
    enc = OneHotEncoder()
    enc.fit(self.x)
    x_transform1 = enc.transform(self.c)
    inverse_transform1 = enc.inverse_transform(x_transform1)
    print(f"times (minilearn) : {round(time.monotonic() - start,4)}")

    start = time.monotonic()
    enc = preprocessing.OneHotEncoder()
    enc.fit(self.x)
    x_transform2 = enc.transform(self.c).toarray()
    inverse_transform2 = enc.inverse_transform(x_transform2)
    print(f"times (sklearn) : {round(time.monotonic() - start,4)}")

    np.testing.assert_array_equal(inverse_transform1,inverse_transform2)
    


if __name__ == "__main__":
  unittest.main()
