import unittest
from minilearn.datasets import read_csv
from minilearn.pipeline import ColumnTransformer
from minilearn.encoders import OneHotEncoder,OrdinalEncoder,TargetEncoder,LabelEncoder
from sklearn import preprocessing,compose
import numpy as np
import pandas as pd
import time

class TestColumnTransformer(unittest.TestCase):

  def setUp(self) -> None:
    df = read_csv("examples/dataset/drug200.csv")
    self.df = df
    data = df.data          # Data -> ('Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K', 'Drug')
    self.X = data[:,:-1]    # X -> ('Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'
    self.y = data[:,-1]     # y -> ("Drug")
    self.remainder = "passthrough"
    smooth = 2
    self.t  = lambda x,z,y,:  [
      # ("OrdinalEncoder",x(),[1]),
      ("OneHotEncoder",x(),[1]),
      ("OrdinalEncoder",z(),[2]),
      ("TargetEncoder",y(smooth=smooth),[3]),

    ]
  
  def test_attribute(self):
    cl1 = ColumnTransformer(self.t(OneHotEncoder,OrdinalEncoder,TargetEncoder),remainder=self.remainder).fit(self.X,self.y)

    cl2 = compose.ColumnTransformer(self.t(preprocessing.OneHotEncoder,preprocessing.OrdinalEncoder,preprocessing.TargetEncoder),remainder=self.remainder).fit(self.X,self.y)
    self.assertDictEqual(cl1.output_indices_,cl2.output_indices_)
  
  def test_transform(self): 
    cl = ColumnTransformer(self.t(OneHotEncoder,OrdinalEncoder,TargetEncoder),remainder=self.remainder).fit(self.X,self.y)
    col_trans1 = cl.transform(self.X).astype(np.float64)

    cl = compose.ColumnTransformer(self.t(preprocessing.OneHotEncoder,preprocessing.OrdinalEncoder,preprocessing.TargetEncoder),remainder=self.remainder).fit(self.X,self.y)
    col_trans2 = cl.transform(self.X).astype(np.float64)
    np.testing.assert_allclose(col_trans1,col_trans2)
  
  def test_performance(self):
    start = time.monotonic()
    cl = ColumnTransformer(self.t(OneHotEncoder,OrdinalEncoder,TargetEncoder),remainder=self.remainder).fit(self.X,self.y)
    col_trans1 = cl.transform(self.X)
    times1 = time.monotonic() - start
    start =  time.monotonic()
    cl = compose.ColumnTransformer(self.t(preprocessing.OneHotEncoder,preprocessing.OrdinalEncoder,preprocessing.TargetEncoder),remainder=self.remainder).fit(self.X,self.y)
    col_trans2 = cl.transform(self.X)
    times2 = time.monotonic() - start
    self.assertLess(times1,times2)

  def test_fit_transform_from_dataset(self):
    columns = self.df.columns
    self.df.fit(features=columns[:-1],target=columns[-1])
    t = [
        ("OneHotEncoder","OneHotEncoder",["Sex"]),
        ("Ordinal","OrdinalEncoder",["BP"]),
        ("TargetEncoder",TargetEncoder(smooth=2),["Cholesterol"])
    ]

    cl = ColumnTransformer(t,remainder=self.remainder).fit_from_dataset(self.df)
    t1 = cl.transform(self.df).astype(np.float64)
    df = pd.read_csv("examples/dataset/drug200.csv")
    X = df[list(columns[:-1])]
    y = df[columns[-1]]

    t = [
        ("OneHotEncoder",preprocessing.OneHotEncoder(),["Sex"]),
        ("Ordinal",preprocessing.OrdinalEncoder(),["BP"]),
        ("TargetEncoder",preprocessing.TargetEncoder(smooth=2),["Cholesterol"])
    ]
    cl = compose.ColumnTransformer(t,remainder=self.remainder).fit(X,y)
    t2 = cl.transform(X).astype(np.float64)
    np.testing.assert_allclose(t1,t2)





if __name__ == "__main__":
  unittest.main()

