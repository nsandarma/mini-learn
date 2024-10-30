import unittest

from pandas.core.interchange.dataframe_protocol import Column

from minilearn.datasets import read_csv
from minilearn.pipeline import ColumnTransformer
from minilearn.encoders import OneHotEncoder,OrdinalEncoder,TargetEncoder,LabelEncoder
from sklearn import preprocessing,compose

import numpy as np
import pandas as pd
import time


class TestColumnTransformer(unittest.TestCase):
  def setUp(self) -> None:
    ds = read_csv("examples/dataset/drug200.csv")
    self.ds = ds
    self.df = pd.read_csv("examples/dataset/drug200.csv")
    data = ds.values
    self.X = data[:,:-1]
    self.y = data[:,-1]
    self.remainder = "passthrough"
    self.cols = [
        [
          [1],[2],[3]
          ]
        ,
        [
        ["Sex"],["BP"],["Cholesterol"]
          ]
        ]
    self.t = lambda x,y,z,idx : [
        ("OneHotEncoder",x(),self.cols[idx][0]),
        ("OrdinalEncoder",y(),self.cols[idx][1]),
        ("TargetEncoder",z(smooth=2),self.cols[idx][2])
        ]


  def test_from_array(self):

    t1 = self.t(OneHotEncoder,OrdinalEncoder,TargetEncoder,0)
    t2 = self.t(preprocessing.OneHotEncoder,preprocessing.OrdinalEncoder,preprocessing.TargetEncoder,0)

    cl1 = ColumnTransformer(t1,remainder=self.remainder).fit(self.X,self.y)

    cl2 = compose.ColumnTransformer(t2,remainder=self.remainder).fit(self.X,self.y)
    np.testing.assert_allclose(cl1.transform(self.X).astype(np.float64),
                               cl2.transform(self.X).astype(np.float64))
    self.assertDictEqual(cl1.output_indices_,cl2.output_indices_)

  def test_dtype(self):
    t1 = self.t(OneHotEncoder,OrdinalEncoder,TargetEncoder,1)
    t2 = self.t(preprocessing.OneHotEncoder,preprocessing.OrdinalEncoder,preprocessing.TargetEncoder,1)

    features = self.df.columns[:-1].tolist()
    target =  "Drug"
    X = self.ds.get(features)
    y = self.ds.get(target)
    cl1 = ColumnTransformer(t1,remainder=self.remainder).fit(X,y)

    # X = self.df[features]
    # y = self.df[target]
    # cl1 = compose.ColumnTransformer(t2,remainder=self.remainder).fit(X,y)

  def test_from_dataset(self):

    t1 = self.t(OneHotEncoder,OrdinalEncoder,TargetEncoder,1)
    t2 = self.t(preprocessing.OneHotEncoder,preprocessing.OrdinalEncoder,preprocessing.TargetEncoder,1)
    features = self.df.columns[:-1].tolist()
    target =  "Drug"

    cl1 = ColumnTransformer(t1,remainder=self.remainder).fit(self.ds.get(features),self.ds.get(target))
    cl2 = compose.ColumnTransformer(t2,remainder=self.remainder).fit(self.df[features],self.df[target])

    np.testing.assert_allclose(cl1.transform(self.ds.get(features)).astype(np.float64),
                               cl2.transform(self.df[features]).astype(np.float64))

    self.assertDictEqual(cl1.output_indices_,cl2.output_indices_)

  def test_performance(self):

    t1 = self.t(OneHotEncoder,OrdinalEncoder,TargetEncoder,0)
    t2 = self.t(preprocessing.OneHotEncoder,preprocessing.OrdinalEncoder,preprocessing.TargetEncoder,0)
    start = time.monotonic()
    cl1 = ColumnTransformer(t1,remainder=self.remainder).fit(self.X,self.y)
    cl1.transform(self.X)
    t1 = time.monotonic() - start

    start = time.monotonic()
    cl2 = compose.ColumnTransformer(t2,remainder=self.remainder).fit(self.X,self.y)
    cl2.transform(self.X)
    t2 = time.monotonic() - start
    self.assertLess(t1,t2)


    features = self.df.columns[:-1].tolist()
    target = "Drug"

    t1 = self.t(OneHotEncoder,OrdinalEncoder,TargetEncoder,1)
    t2 = self.t(preprocessing.OneHotEncoder,preprocessing.OrdinalEncoder,preprocessing.TargetEncoder,1)
    start = time.monotonic()
    cl1 = ColumnTransformer(t1,remainder=self.remainder).fit(self.ds.get(features),self.ds.get(target))
    cl1.transform(self.ds.get(features))
    t1 = time.monotonic() - start

    start = time.monotonic()
    cl2 = compose.ColumnTransformer(t2,remainder=self.remainder).fit(self.df[features],self.df[target])
    cl2.transform(self.df[features])
    t2 = time.monotonic() - start
    self.assertLess(t1,t2)



if __name__ == "__main__":
  unittest.main()


