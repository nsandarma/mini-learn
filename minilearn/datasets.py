import numpy as np
from numpy import dtype
import os
from typing import Union

def read_csv(path, delimiter=",", return_dataset=True):
  # Memastikan file ada
  assert os.path.exists(path), f"{path} not found!"
  
  # Membaca CSV menggunakan genfromtxt, mengisi nilai kosong dengan np.nan
  data = np.genfromtxt(path,
   delimiter=delimiter, 
   dtype = None,
   encoding=None, 
   names=True, 
   missing_values='', 
   filling_values=0,
  )
  
  columns = data.dtype.names

  # NOTES: columns_dtypes -> list

  column_dtypes = [(np.dtype("int64") if np.issubdtype(data.dtype[name], np.integer) else
                          np.dtype("float64") if np.issubdtype(data.dtype[name], np.floating) else
                          np.dtype("O"))
                   for name in columns]

  data_ = np.array([tuple(row.item()) for row in data], dtype='object')
  data_[data_ == ''] = np.nan
  
  if return_dataset:
    return Dataset(data=data_, columns=columns, dtypes=column_dtypes)

  return data_, columns

class Dataset():

  def __str__(self):
    n_rows,n_cols = self.shape
    missing_values = sum(self.isna().values())
    text = f"<Dataset rows : {n_rows} | columns : {n_cols}>"
    if missing_values:
      text += f" | missing values : {missing_values}"
    return text

  def __repr__(self):
    return str(self)

  def __init__(self,data:np.ndarray,columns:tuple,dtypes=None):
    self.__data = data
    self.__columns = tuple(columns) if  not isinstance(columns,tuple)  else columns
    self.__dtypes = tuple(dtypes) if not isinstance(dtypes,tuple) else dtypes

    self.n_samples,self.n_columns = data.shape

  def __getitem__(self,indexer):
    if isinstance(indexer,(np.ndarray,list,tuple,set)) and all(isinstance(i,str) for i in indexer): 
      indexer = [self.columns.index(i) for i in indexer]
      dtype_selected = [self.__dtypes[i] for i in indexer]
      if np.dtype("object") in dtype_selected:
        dtype = np.dtype("object")
      elif np.dtype("float") in dtype_selected:
        dtype = np.dtype("float")
      else:
        dtype = np.dtype("int")
      return self.values[:,indexer].astype(dtype)
    else: 
      if isinstance(indexer,str): 
        idx = self.columns.index(indexer)
        return self.values[:,idx].astype(self.dtypes[indexer])
      return self.values[indexer]

  def get(self,indexer):
    if isinstance(indexer,(np.ndarray,list,tuple,set)) and all(isinstance(i,str) for i in indexer): 
      idx = [self.columns.index(i) for i in indexer]
      data = self.values[:,idx]
      dtypes = [self.__dtypes[i] for i in idx]
      return Dataset(data,indexer,dtypes)
    else: 
      if isinstance(indexer,str): 
        idx = self.columns.index(indexer)
        data = self.values[:,[idx]].astype(self.dtypes[indexer])
        dtypes = [self.dtypes[indexer]]
        return Dataset(data,[indexer],dtypes)
      data = self[indexer]
      return Dataset(data,self.columns,self.__dtypes)

  def select_dtypes(self,include=None,exclude=None):
    dtype = np.dtype(include) if include else np.dtype(exclude)
    if include:
      dtype_selected = [i for i,v in enumerate(self.__dtypes) if v == dtype]
    else:
      dtype_selected = [i for i,v in enumerate(self.__dtypes) if v != dtype]

    if dtype_selected:
      col_selected = np.array(self.columns)[dtype_selected].tolist()
      return self.get(col_selected)
    return []

  
  def head(self,samples=5):
    samples = abs(samples)
    if samples > self.n_samples: samples = self.n_samples
    return self[:samples]

  def tail(self,samples=5):
    if samples > 0 : samples = -samples
    if samples < -self.n_samples: samples = -self.n_samples
    return self[samples:]
  
  def isna(self,return_counts=True):
    data_ = self.values
    # Menggunakan np.vectorize untuk memeriksa np.nan
    is_nan = np.vectorize(lambda x: isinstance(x, float) and np.isnan(x))
    is_nan = is_nan(data_)
    if return_counts:
      result = {}
      for idx,col in enumerate(self.columns):
        result[col] = sum(is_nan[:,idx]).item()
      return result
    return is_nan


  def drop(self,index=None,columns=None,axis=0,inplace=False):
    axis = 1 if columns  else axis
    if axis == 1:
      assert isinstance(columns,(list,np.ndarray,tuple,set)), "type(columns) !"
      index = np.isin(self.columns,columns) 
      columns = np.delete(self.columns,index,0)
      dtypes = np.delete(self.__dtypes,index,0)

      data = np.delete(self.values,index,1)

      if inplace:
        self.__columns = columns
        self.__dtypes = dtypes
        self.__data = data
        return 
      return Dataset(data,columns,dtypes)

    else:
      assert isinstance(index,(list,np.ndarray,tuple,set)), "type(columns) !"
      data = np.delete(self.values,index,0)
      if inplace:
        self.__data = data
        return 
      return Dataset(data,self.columns,self.__dtypes)


  def dropna(self,axis=0,inplace=False):
    is_nan = self.isna(return_counts=axis)
    if axis == 1:
      cols_nan = [cols for cols,values in is_nan.items() if values != 0]
      assert cols_nan , "no missing values!"
      return self.drop(columns=cols_nan,inplace=inplace)
    else:
      idx_nan = np.where(is_nan)[0]
      return self.drop(index=idx_nan,axis=0,inplace=inplace)


  def save(self,filename):
    with open(filename,"wb") as f:
      np.save(f,self.values)
  
  def save_obj(self,filename):
    pass
  
  @property
  def values(self):
    if np.dtype("object") in self.__dtypes:
      dtype = np.dtype("object")
    elif np.dtype("float") in self.__dtypes:
      dtype = np.dtype("float")
    else:
      dtype = np.dtype("int")
    return self.__data.astype(dtype)
  
  @property
  def columns(self):
    return self.__columns

  @property
  def dtype(self):
    return self.values.dtype

  @property
  def dtypes(self):
    return {col:dtype for col,dtype in zip(self.columns,self.__dtypes)}
  
  @property
  def shape(self):
    return self.values.shape
  

# generate_synthetic_data

def make_classification(n_samples=100,n_features=2,n_classes=2,mean_range=(0,5),std=1.0,random_state=None):
  if random_state is not None:
    np.random.seed(random_state)
  samples_per_class = n_samples // n_classes
  X = []
  y = []
  
  for class_label in range(n_classes):
    mean = np.random.uniform(mean_range[0], mean_range[1], n_features)
    X_class = np.random.normal(loc=mean, scale=std, size=(samples_per_class, n_features))
    y_label = np.full(samples_per_class,class_label)
    X.append(X_class)
    y.append(y_label)
  X = np.vstack(X)
  y = np.hstack(y)
  indices = np.random.permutation(n_samples)
  return X[indices],y[indices]


if __name__ == "__main__":
  X,y = make_classification(n_features=5,n_classes=2,random_state=42)
