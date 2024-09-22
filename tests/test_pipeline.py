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
    data = df.data          # Data -> ('Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K', 'Drug')
    self.X = data[:,:-1]    # X -> ('Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'
    self.y = data[:,-1]     # y -> ("Drug")
    self.remainder = "passthrough"
    smooth = 2
    self.t  = lambda x,z,y,:  [
      ("OrdinalEncoder",x(),[1]),
      ("OneHotEncoder",z(),[2]),
      ("TargetEncoder",y(smooth=smooth),[3]),
    ]
  
  def test_attribute(self):
    start =  time.monotonic()
    cl = ColumnTransformer(self.t(OrdinalEncoder,OneHotEncoder,TargetEncoder),remainder=self.remainder).fit(self.X,self.y)

    start =  time.monotonic()
    cl = compose.ColumnTransformer(self.t(preprocessing.OrdinalEncoder,preprocessing.OneHotEncoder,preprocessing.TargetEncoder),remainder=self.remainder).fit(self.X,self.y)
    print(cl.output_indices_)

  def test_transform(self): 
    start =  time.monotonic()
    cl = ColumnTransformer(self.t(OrdinalEncoder,OneHotEncoder,TargetEncoder),remainder=self.remainder).fit(self.X,self.y)
    col_trans1 = cl.transform(self.X).astype(np.float64)
    print(f"times : {time.monotonic() - start}")

    start =  time.monotonic()
    cl = compose.ColumnTransformer(self.t(preprocessing.OrdinalEncoder,preprocessing.OneHotEncoder,preprocessing.TargetEncoder),remainder=self.remainder).fit(self.X,self.y)
    col_trans2 = cl.transform(self.X).astype(np.float64)
    np.testing.assert_allclose(col_trans1,col_trans2)
    print(f"times : {time.monotonic() - start}")



if __name__ == "__main__":
  unittest.main()

    
    

  
    

