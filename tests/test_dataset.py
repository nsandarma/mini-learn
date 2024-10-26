import unittest
from minilearn.datasets import Dataset,read_csv
import pandas as pd
import numpy as np

class TestDataset(unittest.TestCase):
  def setUp(self):
    self.p_drug = "examples/dataset/drug200.csv"
    self.dataset = read_csv(self.p_drug)
    self.dataframe = pd.read_csv(self.p_drug)
    
  def test_dtypes(self):
    dtypes_df = self.dataframe.dtypes.values
    dtypes= self.dataset.dtypes
    print(dtypes_df)

  def test_getitem(self):
    cols_selected = ["Sex","Age"]
    ds_cols_selected = self.dataset[cols_selected]
    df_cols_selected = self.dataframe[cols_selected].values
    np.testing.assert_equal(ds_cols_selected,df_cols_selected)


    ds_idx_selected = self.dataset[1:5]
    df_idx_selected = self.dataframe.iloc[1:5].values
    np.testing.assert_equal(ds_idx_selected,df_idx_selected)


if __name__ == "__main__":
  unittest.main()
