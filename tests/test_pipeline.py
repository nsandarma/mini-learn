import unittest
from minilearn.datasets import read_csv
from minilearn.pipeline import ColumnTransformer
from minilearn.encoders import OneHotEncoder,OrdinalEncoder,TargetEncoder,LabelEncoder
from sklearn import preprocessing,compose
import numpy as np
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
      ("OneHotEncoder",z(),[1,2]),
      ("TargetEncoder",y(smooth=smooth),[3]),
    ]
  
  def test_attribute(self):
    cl1 = ColumnTransformer(self.t(OrdinalEncoder,OneHotEncoder,TargetEncoder),remainder=self.remainder).fit(self.X,self.y)

    cl2 = compose.ColumnTransformer(self.t(preprocessing.OrdinalEncoder,preprocessing.OneHotEncoder,preprocessing.TargetEncoder),remainder=self.remainder).fit(self.X,self.y)
    self.assertDictEqual(cl1.output_indices_,cl2.output_indices_)
  
  def test_anyway(self):
    t = [
      ("onehot","OneHotEncoder",["s","b",1]),
      ("ordinal","OrdinalEncoder",["a","x",2]),
    ]
    r = "drop"
    cl  = ColumnTransformer(t,remainder=self.remainder)

  def test_transform(self): 
    cl = ColumnTransformer(self.t(OrdinalEncoder,OneHotEncoder,TargetEncoder),remainder=self.remainder).fit(self.X,self.y)
    col_trans1 = cl.transform(self.X).astype(np.float64)

    cl = compose.ColumnTransformer(self.t(preprocessing.OrdinalEncoder,preprocessing.OneHotEncoder,preprocessing.TargetEncoder),remainder=self.remainder).fit(self.X,self.y)
    col_trans2 = cl.transform(self.X).astype(np.float64)
    np.testing.assert_allclose(col_trans1,col_trans2)
  
  def test_performance(self):
    start = time.monotonic()
    cl = ColumnTransformer(self.t(OrdinalEncoder,OneHotEncoder,TargetEncoder),remainder=self.remainder).fit(self.X,self.y)
    col_trans1 = cl.transform(self.X)
    times1 = time.monotonic() - start
    start =  time.monotonic()
    cl = compose.ColumnTransformer(self.t(preprocessing.OrdinalEncoder,preprocessing.OneHotEncoder,preprocessing.TargetEncoder),remainder=self.remainder).fit(self.X,self.y)
    col_trans2 = cl.transform(self.X)
    times2 = time.monotonic() - start
    self.assertLess(times1,times2)


if __name__ == "__main__":
  unittest.main()

    
    

  
    

