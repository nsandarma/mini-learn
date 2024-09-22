import numpy as np
from typing import Literal
from minilearn.encoders import ENCODERS,OneHotEncoder,OrdinalEncoder,TargetEncoder,LabelEncoder
from collections import OrderedDict

class Pipeline:
  def __init__(self):...


class ColumnTransformer:
  def __encoder(self,encoder:Literal["OneHotEncoder","OrdinalEncoder","TargetEncoder","LabelEncoder"]):
    assert encoder in ENCODERS, f"{encoder} not in list encoders : {ENCODERS}" 
    if encoder == "OneHotEncoder":
      return OneHotEncoder()
    elif encoder == "OrdinalEncoder":
      return OrdinalEncoder()
    elif encoder == "TargetEncoder":
      return TargetEncoder()
    else:
      return LabelEncoder()
  
  def __columns_validate(self,X):
    columns_ = self.__columns
    if not self.__cols_spec:
      assert np.max(columns_) <= X.shape[1] and np.min(columns_) >= 0
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
      if isinstance(process,str): process = self.__encoder(process)
      else:
        assert isinstance(process,object)
      columns_.append(columns)
      n_process_.append(n_process)
      process_.append(process)
    self.__n_process = n_process_
    self.__process = process_
    self.__cols_spec = any(isinstance(i,str) for col in columns_ for i in col) # specifying the columns
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

  def __init__(self,transformers:list,remainder:Literal["drop","passthrough"]="drop",verbose=False):
    # format : 
    # [ (process_name,process,column_names)
    # ]
    assert remainder in ["drop","passthrough"] , f'remainder : {remainder} is not found in ["drop","passthrough"]'
    self.__check_transformers(transformers=transformers)
    self.transformers = transformers
    self.remainder = remainder
    self.is_fitted = False
    self.verbose = verbose
  
  def __check_remainder(self,idx_col_last,diff):
    if self.remainder == "passthrough":
      return slice(idx_col_last,idx_col_last+diff)
    return slice(0,0)


  def fit(self,X,y=None): 
    X,y = np.asarray(X),np.asarray(y)
    assert not self.__cols_spec,"Specifying the columns using strings is only supported for dataset. use `fit_from_dataset`"
    self.n_cols = X.shape[1]
    columns = self.__columns
    n_process = self.__n_process
    self.__output_indices = dict()
    idx = 0
    for i,n in enumerate(n_process):
      process = self.__process[i]
      X_c = X[:,columns[i]]
      if isinstance(process,TargetEncoder):
        assert y is not None, "y tidak boleh kosong!"
        process.fit(X_c,y)
      else:
        if isinstance(process,LabelEncoder):
          X_c = X_c.reshape(-1)
        process.fit(X_c)
      # self.__col_out.append(self.__check_col_out(process,columns[i]))
      col_out = self.__check_col_out(process,columns[i])
      self.__output_indices[n] = slice(idx,idx + col_out)
      idx += col_out
      if self.verbose:
        print(f"[ColumnTransformer] ....... ({i+1} of {len(n_process)}) Processing {n},")

    diff = np.setdiff1d(np.arange(X.shape[1]), np.array(self.__all_cols))
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

  def fit_from_dataset(self,dataset):
    pass

  def transform(self,X):
    self.__check_is_fitted()
    X = np.asarray(X)
    columns = self.__columns
    n_rows,_ = X.shape
    X_out = np.empty((n_rows,self.__n_cols),dtype="object")
    for i,n in enumerate(self.__n_process):
      process = self.__process[i]
      X_c = X[:,columns[i]]
      if isinstance(process,LabelEncoder):
        X_c = X_c.reshape(-1)
        X_trans = process.transform(X_c).reshape(-1,1)
      else:
        X_trans = process.transform(X_c)
      X_out[:,self.__output_indices[n]] = X_trans
    X_out[:,self.__output_indices["remainder"]] = X[:,self.__diff]
    return X_out

  @property
  def process(self):
    return self.__process

