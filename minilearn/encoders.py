import numpy as np
from abc import ABC,abstractmethod

class Encoders:
  is_fitted = False
  t = None

  def fit(self,x):
    x = np.asarray(x)
    if self.t == "features": 
      assert x.ndim != 1 , \
        "Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample."
      self.__categories = [np.unique(x[:,i]) for i in range(x.shape[1])]
      self.n_features = x.shape[1] # origin
      c = 0
      idx_col = []
      n_features_ = []
      for i in range(x.shape[1]):
        n = len(self.categories_[i])
        idx_col.append(range(c,c+n));
        c += n
        n_features_.append(len(self.categories_[i]))
      self.n_features_ = n_features_ # pre encoding
      self.idx_cols = idx_col
    else:
      assert x.ndim == 1 , f"should be a 1d array, got an array of shape {x.shape} instead."
      self.categories_ = np.unique(x)

    self.is_fitted = True
    self.__dim = x.ndim
    return self

  def check_is_fitted(self):
    assert self.is_fitted,f"{self.name} instance is not fitted yet. Call 'fit' with appropriate arguments before using this"
    return 
  
  def check_isin(self,x):
    if self.t != "features":
      assert all(i in self.categories_ for i in np.unique(x)),f"value {np.unique(x)} != {self.categories_}"
    else:
      assert x.ndim == 2, "Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample."
      assert x.shape[1] == self.n_features,f"X has {x.shape[1]} features, but OneHotEncoder is expecting {self.n_features} features as input."
      for i in range(x.shape[1]): 
        x_unique = np.unique(x[:,i])
        isNotmember = [j for j in range(len(x_unique)) if x_unique[j] not in self.categories_[i]]
        assert not isNotmember, f"Found unknown categories {x_unique[isNotmember]} in column {i} during transform"
    return 

  @property
  def categories_(self):
    return self.__categories
  @property
  def dim(self):
    return self.__dim

class OneHotEncoder(Encoders):
  t = "features"
  def __init__(self):
    self.__identity = None
    self.idx_cols = None

  def fit(self,x:np.ndarray):
    super().fit(x)
    self.__identity = [np.eye(len(cat)) for cat in self.categories_]

  def transform(self,x):
    x = np.asarray(x)
    self.check_is_fitted()
    self.check_isin(x)

    result = np.zeros((x.shape[0], sum(len(cat) for cat in self.categories_)), dtype=int)

    for i in range(x.shape[1]):
      cat_indices = np.searchsorted(self.categories_[i], x[:, i])
      identity_rows = self.__identity[i][cat_indices]
      result[:, self.idx_cols[i]] = identity_rows
    return result
    
  def fit_transform(self,x):
    self.fit(x)
    return self.transform(x)
  
  def inverse_transform(self,x):
    self.check_is_fitted()
    assert self.idx_cols, "using transform with encoders minilearn !"
    result = np.empty((x.shape[0],self.n_features),dtype=object)
    for i,cat in enumerate(self.categories_):
      res = cat[np.argmax(x[:,self.idx_cols[i]], axis=1)]
      result[:, i] = res
    return result

  @property
  def identity_(self):
    return self.__identity

class OrdinalEncoder(Encoders):
  t = "features"
  def __init__(self): ...

  def fit(self,x):
    super().fit(x)

  def transform(self,x):
    x = np.asarray(x)
    self.check_is_fitted()
    self.check_isin(x)
    result  = np.zeros((x.shape[0],x.shape[1]),dtype=int)
    for i in range(x.shape[1]):
      cat_indices = np.searchsorted(self.categories_[i],x[:,i])
      result[:,i] = cat_indices
    return result
  
  def fit_transform(self,x):
    self.fit(x)
    return self.transform(x)
  
  def inverse_transform(self,x):
    self.check_is_fitted()
    result = np.empty((x.shape[0],x.shape[1]),dtype=object)
    for i,cat in enumerate(self.categories_):
      res = cat[x[:,i]]
      result[:,i] = res
    return result
    


class LabelEncoder(Encoders):
  t = "label"
  def fit(self,x):
    super().fit(x)
  
  def transform(self,x):
    self.check_is_fitted()
    self.check_isin(x)
    res = np.array([np.where(self.categories_ == i)[0][0] for i in x])
    return res if self.dim == 1 else res.reshape(-1,1)
  
  def fit_transform(self,x):
    self.fit(x)
    return self.transform(x)
  
  def inverse_transform(self,x):
    self.check_is_fitted()
    res = np.array([self.categories_[np.where(self.categories_ == i)[0][0]] for i in x])
    return res if self.dim == 1 else res.shape(-1,1)
  
def onehot(x): return OneHotEncoder().fit_transform(x)
def label(x): return LabelEncoder().fit_transform(x)

if __name__ == "__main__":
  from sklearn import preprocessing
  import pandas as pd
  df = pd.read_csv("examples/dataset/drug200.csv")
  cats = df.select_dtypes("object").values
  s = cats[np.random.randint(0,len(cats),(10))]
  enc = OrdinalEncoder()
  enc.fit(cats)
  x_transform =  enc.transform(s)
  print(x_transform)
  print(enc.inverse_transform(x_transform))
  print()
    
  enc = preprocessing.OrdinalEncoder()
  enc.fit(cats)
  x_transform =  enc.transform(s)
  print(x_transform)
  print(enc.inverse_transform(x_transform))
