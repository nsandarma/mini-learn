import unittest
import pandas as pd
from sklearn.utils import multiclass
from minilearn.utils import type_of_target


class TestUtility(unittest.TestCase):
  def setUp(self) -> None:
    self.df = pd.read_csv("examples/dataset/drug200.csv")
  
  def test_type_of_target(self):
    cols = self.df.columns
    res1 = [type_of_target(self.df[col].values) for col in cols]
    res2 = [multiclass.type_of_target(self.df[col].values) for col in cols]
    print(res1)
    print(res2)


if __name__ == "__main__":
  unittest.main()





