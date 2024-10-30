import unittest
from minilearn.datasets import Dataset,read_csv
import pandas as pd
import numpy as np

def set_missing_values(dataframe,nums):
  rows,cols = dataframe.values.shape
  for i in range(nums):
    idx = np.random.randint(0,rows)
    col = np.random.randint(0,cols)
    dataframe.iloc[idx,col] = np.nan
  

class TestDataset(unittest.TestCase):
  def setUp(self):
    self.p_drug = "examples/dataset/drug200.csv"
    self.dataset = read_csv(self.p_drug)
    self.dataframe = pd.read_csv(self.p_drug)
    
  def test_dtypes(self):
    dtypes_df = self.dataframe.dtypes.to_dict()
    dtypes_ds = self.dataset.dtypes
    self.assertDictEqual(dtypes_df,dtypes_ds)

    dd_ds = []
    dd_df = []

    for col in self.dataset.columns:
      dd_ds.append(self.dataset[col].dtype)
      dd_df.append(self.dataframe[col].dtype)
    np.testing.assert_array_equal(dd_ds,dd_df)

    ds_selected = self.dataset.select_dtypes("object")
    df_selected = self.dataframe.select_dtypes("object")
    self.assertDictEqual(ds_selected.dtypes,df_selected.dtypes.to_dict())

    ds_selected = self.dataset.select_dtypes(exclude="object")
    df_selected = self.dataframe.select_dtypes(exclude="object")
    self.assertDictEqual(ds_selected.dtypes,df_selected.dtypes.to_dict())

    for col in ds_selected.columns:
      self.assertEqual(ds_selected[col].dtype,df_selected[col].dtype)

  def test_dtype(self):
    ds = self.dataset.select_dtypes(exclude="object")
    df = self.dataframe.select_dtypes(exclude="object")
    self.assertEqual(ds.dtype,df.values.dtype)
    

  def test_getitem(self):
    cols_selected = ["Sex","Age"]
    ds_cols_selected = self.dataset[cols_selected]
    df_cols_selected = self.dataframe[cols_selected].values
    np.testing.assert_array_equal(ds_cols_selected,df_cols_selected)

    ds_idx_selected = self.dataset[1:5]
    df_idx_selected = self.dataframe.iloc[1:5].values
    np.testing.assert_equal(ds_idx_selected,df_idx_selected)

    idx_selected = [1,2,3,4,5]
    ds_idx_selected = self.dataset[idx_selected]
    df_idx_selected = self.dataframe.values[idx_selected]
    np.testing.assert_array_equal(ds_idx_selected,df_idx_selected)

  def test_isna(self):
    set_missing_values(self.dataframe,10)
    df_isna = self.dataframe.isna().sum().to_dict()

    data = self.dataframe.values
    cols = self.dataframe.columns.tolist()
    dtypes = self.dataframe.dtypes.tolist()

    ds = Dataset(data=data,columns=cols,dtypes=dtypes)
    self.assertDictEqual(df_isna,ds.isna())

  def test_drop(self):
    col_selected = ["Sex","Drug"]
    idx_selected = [0,2,4]
    ds_remove = self.dataset.drop(columns=col_selected,inplace=False)
    df_remove = self.dataframe.drop(columns=col_selected,inplace=False)
    np.testing.assert_array_equal(ds_remove.values,df_remove.values)

    ds_remove = self.dataset.drop(index=idx_selected,axis=0,inplace=False)
    df_remove = self.dataframe.drop(index=idx_selected,axis=0,inplace=False)
    np.testing.assert_array_equal(ds_remove.values,df_remove.values)

  def test_get(self):
    # by column names
    cols_selected = ["Sex","Age","Na_to_K"]
    ds = self.dataset.get(cols_selected)
    np.testing.assert_array_equal(ds.columns,cols_selected)
    dty = {col:self.dataset.dtypes[col] for col in cols_selected}
    self.assertDictEqual(dty,ds.dtypes)
    np.testing.assert_array_equal(self.dataset[cols_selected],ds.values)

    # by index
    idx_selected = [1,2,3,4]
    ds = self.dataset.get(idx_selected)
    np.testing.assert_array_equal(ds.columns,self.dataset.columns)
    self.assertDictEqual(ds.dtypes,self.dataset.dtypes)
    np.testing.assert_array_equal(ds.values,self.dataset[idx_selected])

  def test_select_dtypes(self):
    dtype_selected = np.dtype("object")
    ds_selected = self.dataset.select_dtypes(exclude=dtype_selected)
    df_selected = self.dataframe.select_dtypes(exclude=dtype_selected)
    np.testing.assert_array_equal(ds_selected.columns,df_selected.columns)
    np.testing.assert_array_equal(ds_selected.values,df_selected.values)
    self.assertDictEqual(ds_selected.dtypes,df_selected.dtypes.to_dict())

    


if __name__ == "__main__":
  unittest.main()
