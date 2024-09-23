import unittest
from minilearn.datasets import Dataset,read_csv
import pandas as pd
import numpy as np
import time

class TestDataset(unittest.TestCase):
  def setUp(self):
    path = "examples/dataset/drug200.csv"
    self.dataset = read_csv(path)
    self.dataframe = pd.read_csv(path)
    

  def test_dtypes(self):
    dtypes_df = self.dataframe.dtypes.values
    dtypes= self.dataset.dtypes
    print(dtypes_df)

  def test_getitem(self):
    cols_selected = ["Sex","BP"]
    ta = "Drug"
    start  = time.monotonic()
    self.dataset[cols_selected]
    end = round(time.monotonic() - start,5)
    print(f"times : {end}")

    start  = time.monotonic()
    self.dataframe[cols_selected]
    end = round(time.monotonic() - start,5)
    print(f"times : {end}")

    




if __name__ == "__main__":
  unittest.main()
