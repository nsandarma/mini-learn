import numpy as np
from abc import ABC,abstractmethod
from typing import Literal,Union
from minilearn.utils import type_of_target
from collections import OrderedDict

class Encoders:
  is_fitted = False
  def fit(self,x):
    x = np.asarray(x)
    if str(self) in ["OneHotEncoder","OrdinalEncoder","TargetEncoder"]: 
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
      self.__categories = None
      self.classes_ = np.unique(x)

    self.is_fitted = True
    self.__dim = x.ndim
    return self

  def check_is_fitted(self):
    assert self.is_fitted,f"{self.name} instance is not fitted yet. Call 'fit' with appropriate arguments before using this"
    return 
  
  def check_isin(self,x):
    if str(self) not in ["OneHotEncoder","OrdinalEncoder","TargetEncoder"]:
      isNotMember = [j for j in np.unique(x) if j not in self.classes_]
      assert not isNotMember, f"Found unknown labels {isNotMember}"
    else:
      assert x.ndim == 2, "Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample."
      assert x.shape[1] == self.n_features,f"X has {x.shape[1]} features, but OneHotEncoder is expecting {self.n_features} features as input."
      for i in range(x.shape[1]): 
        x_unique = np.unique(x[:,i])
        isNotmember = [j for j in range(len(x_unique)) if x_unique[j] not in self.categories_[i]]
        assert not isNotmember, f"Found unknown categories {x_unique[isNotmember]} in column {i} during transform"
    return True

  @property
  def categories_(self):
    return self.__categories
  @property
  def dim(self):
    return self.__dim


class OneHotEncoder(Encoders):
  def __str__(self):
    return "OneHotEncoder"
  def __init__(self):
    self.__identity = None
    self.idx_cols = None

  def fit(self,x:np.ndarray):
    super().fit(x)
    self.__identity = [np.eye(len(cat)) for cat in self.categories_]
    return self

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
  def __str__(self):
    return "OrdinalEncoder"
  def __init__(self): ...

  def fit(self,x):
    super().fit(x)
    return self

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


class TargetEncoder(Encoders):
  def __str__(self):
    return "TargetEncoder"
  
  def __init__(self,smooth=2,target_type:Literal["auto","binary","multiclass"]="auto"):
    self.smooth = smooth
    self.target_type = target_type

  def fit(self,x,y):
    super().fit(x)
    y = np.asarray(y)
    assert len(x) == len(y), "len(x) != len(y)"
    assert y.ndim == 1 ,f"should be a 1d array, got an array of shape {x.shape} instead."
    if self.target_type == "auto": self.target_type = type_of_target(y)
    if self.target_type == "binary":
      label_encoder = LabelEncoder()
      y = label_encoder.fit_transform(y)
      self.classes_ = label_encoder.classes_
    elif self.target_type == "multiclass":
      label_binarizer = LabelBinarizer()
      y = label_binarizer.fit_transform(y)
      self.classes_ = label_binarizer.classes_
    self.target_mean_ = np.mean(y)
    encodings = []
    for i,cat in enumerate(self.categories_):
      temp = x[:,i]
      _,counts = np.unique(temp,return_counts=True)
      cm = np.array([np.mean(y[np.where(temp == v)]).item() for v in cat])
      smoothed_means = (counts * cm + self.smooth * self.target_mean_) / (counts + self.smooth)
      encodings.append(smoothed_means)
    self.encodings_ = encodings
    return self
  
  def transform(self,x):
    x = np.asarray(x)
    self.check_is_fitted()
    self.check_isin(x)
    result = np.zeros((x.shape[0],x.shape[1]))
    for i,cat in enumerate(self.categories_):
      temp = x[:,i]
      result[:,i] = self.encodings_[i][np.searchsorted(cat,temp)]
    return result
  
  def fit_transform(self,x):
    self.fit(x)
    return self.transform(x)
  
  def inverse_transform(self,x):
    self.check_is_fitted()
    result = np.empty((x.shape[0],x.shape[1]),dtype=object)
    for i,cat in enumerate(self.categories_):
      temp = x[:,i]
      idx = [np.where(self.encodings_[i] == v)[0].item() for v in temp]
      result[:,i] = self.categories_[i][idx]
    return result


class LabelEncoder(Encoders):
  def __str__(self):
    return "LabelEncoder"

  def __init__(self): pass

  def fit(self,y):
    super().fit(y)
    return self
  
  def transform(self,y):
    y = np.asarray(y)
    self.check_is_fitted()
    self.check_isin(y)
    return np.searchsorted(self.classes_,y)
  
  def fit_transform(self,y):
    self.fit(y)
    return self.transform(y)
  
  def inverse_transform(self,x):
    self.check_is_fitted()
    return self.classes_[x]


class LabelBinarizer(Encoders):
  def __init__(self,neg_label=0,pos_label=1):
    self.neg_label = neg_label
    self.pos_label = pos_label

  def fit(self,y):
    super().fit(y)
    self.multiclass = len(np.unique(y)) > 2 
    if self.multiclass: self.identity_ = np.eye(len(self.classes_),dtype=int)
    return self

  def transform(self,y):
    self.check_is_fitted()
    self.check_isin(y)
    if self.multiclass:
      return self.identity_[np.searchsorted(self.classes_, y)]
    return np.searchsorted(self.classes_,y).reshape(-1,1)
  
  def fit_transform(self,y):
    self.fit(y)
    return self.transform(y)
  
  def inverse_transform(self,y): return self.classes_[y] if not self.multiclass else self.classes_[np.argmax(y,axis=1)]
    

def onehot_encoder(x): return OneHotEncoder().fit_transform(x)
def label_encoder(x): return np.unique(x),np.array([np.where(np.unique(x) == i)[0] for i in x]).reshape(-1)