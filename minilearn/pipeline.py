import numpy as np
from typing import Literal
from minilearn.encoders import FEATURE_ENCODER,OneHotEncoder,OrdinalEncoder,TargetEncoder,LabelEncoder
from collections import OrderedDict
from minilearn.datasets import Dataset

class Pipeline:
  def __init__(self):...


class ColumnTransformer:
  def __encoder(self,encoder):
    if isinstance(encoder,str):
      assert encoder in FEATURE_ENCODER, f"{encoder} not in feature_encoders : {FEATURE_ENCODER}" 
      if encoder == "OneHotEncoder":
        return OneHotEncoder()
      elif encoder == "OrdinalEncoder":
        return OrdinalEncoder()
      else:
        return TargetEncoder()
    else:
      assert isinstance(encoder,(OneHotEncoder,OrdinalEncoder,TargetEncoder)), "encoder not support !"
      return encoder

  
  def __columns_validate(self,X):
    cols_t = self.__all_cols
    if isinstance(X,Dataset):
      all_cols = X.columns
    else:
      all_cols = np.arange(X.shape[1])
    diff = np.setdiff1d(cols_t,all_cols)
    assert diff.size == 0, f"found unknown columns {diff}"
    return 

  def __check_transformers(self,transformers):
    assert transformers, "transformers is None !"
    columns_ = []
    process_ = []
    n_process_ = []
    for t in transformers:
      assert len(t) == 3 
      n_process,process,columns = t
      assert isinstance(columns,(list,tuple,set)), "columns must sequences"
      process = self.__encoder(process)
      columns_.append(columns)
      n_process_.append(n_process)
      process_.append(process)
    self.__n_process = n_process_
    self.__process = process_
    self.__columns = columns_
    self.__all_cols = list(set([i for col in columns_ for i in col]))
    assert len(n_process_) == len(process_) 
    assert np.sum(np.unique(n_process_,return_counts=True)[1]) == len(np.unique(n_process_)), f"Names provided are not unique: {n_process_}"
  
  def __check_is_fitted(self):
    assert self.is_fitted,f"{self.name} instance is not fitted yet. Call 'fit' with appropriate arguments before using this"
  
  def __check_col_out(self,process,cols):
    if isinstance(process,(TargetEncoder,OneHotEncoder)): 
      return process.n_features_
    return len(cols)

  def __dtype_ret(self,values:np.ndarray):
    if values.dtype == np.object_:
      if any(isinstance(element, str) for element in values.flat):
        return np.dtype('object')
      elif any(isinstance(element, float) for element in values.flat):
        return np.dtype('float')
      else:
        return np.dtype('int')
    return None


  def __init__(self,transformers:list,remainder:Literal["drop","passthrough"]="drop",verbose=False):
    # format transformer: 
    # [ (process_name,process,column_names)
    # ]
    assert remainder in ["drop","passthrough"] , f'remainder : {remainder} is not found in ["drop","passthrough"]'
    
    self.__check_transformers(transformers=transformers) 
    self.transformers = transformers
    self.remainder = remainder
    self.is_fitted = False
    self.verbose = verbose

  def fit(self,X,y=None): 
    if isinstance(X,Dataset) or isinstance(y,Dataset):
      return self.__fit_from_dataset(X,y)
    X,y = np.asarray(X),np.asarray(y)
    self.__columns_validate(X)
    self.n_cols = X.shape[1]
    columns = self.__columns
    n_process = self.__n_process
    self.__output_indices = dict()
    idx = 0
    for i,n in enumerate(n_process):
      process = self.__process[i]
      X_c = X[:,columns[i]]
      if isinstance(process,TargetEncoder):
        process.fit(X_c,y)
      else:
        process.fit(X_c)

      col_out = self.__check_col_out(process,columns[i])
      self.__output_indices[n] = slice(idx,idx + col_out)
      idx += col_out
      if self.verbose:
        print(f"[ColumnTransformer] ....... ({i+1} of {len(n_process)}) Processing {n},")

    diff = np.setdiff1d(np.arange(X.shape[1]), np.array(self.__all_cols))
    self.__dtype = self.__dtype_ret(X[:,diff])
    if self.remainder == "passthrough" and diff.size > 0:
      output_remainder = slice(idx,idx+len(diff))
      size = output_remainder.stop
    else:
      output_remainder = slice(0,0)
      size = idx
    self.__n_cols = size
    self.__diff = diff
    self.__output_indices["remainder"] = output_remainder
    self.is_fitted = True
    return self

  def __fit_from_dataset(self,X:Dataset,y:Dataset=None):
    assert isinstance(X,Dataset) and isinstance(y,Dataset), "X or y != <Dataset>"
    assert y.shape[1] == 1, "y shape unknown"
    self.__columns_validate(X)
    columns = self.__columns
    y = y.values.reshape(-1)
    self.__output_indices = dict()
    idx = 0

    for i,n in enumerate(self.__n_process):
      process = self.__process[i]
      X_c = X[columns[i]]
      if isinstance(process,TargetEncoder):
        assert y is not None,"fit dataset terlebih dahulu!"
        process.fit(X_c,y)
      else:
        process.fit(X_c)
      col_out = self.__check_col_out(process,columns[i])
      self.__output_indices[n] = slice(idx,idx + col_out)
      idx += col_out
      if self.verbose:
        print(f"[ColumnTransformer] ....... ({i+1} of {len(self.__n_process)}) Processing {n},")

    diff = np.setdiff1d(X.columns, np.array(self.__all_cols))
    self.__dtype = self.__dtype_ret(X[diff])
    if self.remainder == "passthrough" and diff.size > 0:
      output_remainder = slice(idx,idx+len(diff))
      size = output_remainder.stop
    else:
      output_remainder = slice(0,0)
      size = idx
    self.__n_cols = size
    self.__diff = diff.tolist()
    self.__output_indices["remainder"] = output_remainder
    self.is_fitted = True
    return self

  def transform(self,X):
    self.__check_is_fitted()
    if isinstance(X,Dataset):
      return self.__transform_from_dataset(X)
    X = np.asarray(X)
    columns = self.__columns
    n_rows,_ = X.shape
    X_out = np.empty((n_rows,self.__n_cols),dtype=self.__dtype)
    for i,n in enumerate(self.__n_process):
      process = self.__process[i]
      X_c = X[:,columns[i]]
      X_trans = process.transform(X_c)
      X_out[:,self.__output_indices[n]] = X_trans
    if self.remainder != "drop": X_out[:,self.__output_indices["remainder"]] = X[:,self.__diff]
    return X_out
  
  def __transform_from_dataset(self,dataset:Dataset):
    columns = self.__columns
    n_rows,_ = dataset.values.shape
    X_out = np.empty((n_rows,self.__n_cols),dtype=self.__dtype)
    for i,n in enumerate(self.__n_process):
      process = self.__process[i]
      X_c = dataset[columns[i]]
      X_trans = process.transform(X_c)
      X_out[:,self.__output_indices[n]] = X_trans
    if self.remainder != "drop": X_out[:,self.__output_indices["remainder"]] = dataset[self.__diff]
    return X_out
  
  def fit_transform(self,X,y=None):
    if isinstance(X,Dataset):
      self.fit_from_dataset(X)
    else:
      self.fit(X,y)
    return self.transform(X)

  @property
  def process(self):
    return self.__process
  
  @property
  def output_indices_(self):
    return self.__output_indices


if __name__ == "__main__":
  from minilearn.datasets import read_csv
  # Data -> ('Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K', 'Drug')
   
  ds = read_csv("examples/dataset/drug200.csv")
  cols = ds.columns
  ds.fit(features = cols[:-1],target=cols[-1])
  X,y = ds.data[:,:-1],ds.data[:,-1]
  smooth = 2
  t = [
      # ("OrdinalEncoder",x(),[1]),
      ("OneHotEncoder",OneHotEncoder(),["Sex"]),
      ("OrdinalEncoder","OrdinalEncoder",["BP"]),
      ("TargetEncoder","TargetEncoder",["Cholesterol"])
  ]
  cl = ColumnTransformer(t,"passthrough").fit_from_dataset(ds)
  print(cl.transform(ds).shape)
