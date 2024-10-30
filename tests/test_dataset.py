import unittest
from minilearn.datasets import Dataset,read_csv
from minilearn.utils import train_test_split
import pandas as pd
import numpy as np

def set_missing_values(dataframe,nums,seed=42):
  rows,cols = dataframe.values.shape
  np.random.seed(seed)
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

  def test_train_test_split(self):
    from sklearn import model_selection
    random_state = 42
    train_size = 0.8
    shuffle = False

    features = self.dataframe.columns[:-1].tolist()
    target = "Drug"
    X = self.dataset.get(features)
    y = self.dataset.get(target)
    X_train,y_train,X_test,y_test = train_test_split(X,y,random_state=random_state,train_size=train_size,shuffle=shuffle)

  
    X_train_,X_test_,y_train_,y_test_ = model_selection.train_test_split(self.dataframe[features],
                                                                         self.dataframe[[target]],
                                                                         shuffle=shuffle,train_size=train_size,
                                                                         random_state=random_state)
    np.testing.assert_array_equal(X_train_,X_train)
    np.testing.assert_array_equal(y_train_,y_train)
    np.testing.assert_array_equal(X_test_,X_test)
    np.testing.assert_array_equal(y_test_,y_test)

    X_train,y_train,X_test,y_test = train_test_split(X,y,random_state=42,train_size=train_size,shuffle=True)
    expected = int(train_size * len(self.dataset))
    self.assertEqual(X_train.shape[0],expected)
    self.assertEqual(y_train.shape[0],expected)
    self.assertEqual(X_test.shape[0],len(self.dataset) - expected)
    self.assertEqual(y_test.shape[0],len(self.dataset) - expected)


  def test_value_counts(self):
    df = self.dataframe[["Sex"]].copy()
    set_missing_values(df,np.random.randint(0,20))
    data = df.values
    cols = df.columns.tolist()
    dtypes = df.dtypes.tolist()
    ds = Dataset(data=data,columns=cols,dtypes=dtypes)
    ds_val = ds.value_counts(dropna=False,normalize=True)
    s = df.value_counts(dropna=False,normalize=True).to_dict()
    key = [str(i[0]) for i in s.keys()]
    s = {k:v for k,v in zip(key,s.values())}
    self.assertDictEqual(ds_val,s)
    


if __name__ == "__main__":
  unittest.main()
