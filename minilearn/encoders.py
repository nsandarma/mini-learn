import numpy as np
from abc import ABC,abstractmethod

class Encoders:
  is_fitted = False
  def fit(self,x):
    x = np.asarray(x)
    if str(self) in ["OneHotEncoder","OrdinalEncoder","TargetEncoder"]: 
      assert x.ndim != 1 , \
        "Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample."
      return_counts = str(self) == "TargetEncoder"
      self.__categories = [np.unique(x[:,i],return_counts=return_counts) for i in range(x.shape[1])]
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
      self.__categories = np.unique(x)

    self.is_fitted = True
    self.__dim = x.ndim
    return self

  def check_is_fitted(self):
    assert self.is_fitted,f"{self.name} instance is not fitted yet. Call 'fit' with appropriate arguments before using this"
    return 
  
  def check_isin(self,x):
    if str(self) not in ["OneHotEncoder","OrdinalEncoder","TargetEncoder"]:
      isNotMember = [j for j in np.unique(x) if j not in self.categories_]
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
  
  def __init__(self,smooth="auto",shuffle=True,random_state=None):
    self.__smooth = smooth
    self.__shuffle = shuffle
    self.__random_state = random_state

  def fit(self,x,y):
    super().fit(x)
    y = np.asarray(y)
    assert len(x) == len(y), "len(x) != len(y)"
    assert y.ndim == 1 ,f"should be a 1d array, got an array of shape {x.shape} instead."
    self.classess_ = np.unique(y,return_counts=True)
    return self
  
  def transform(self,x):
    x = np.asarray(x)
    self.check_is_fitted()
    self.check_isin(x)


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
    return np.searchsorted(self.categories_,y)
  
  def fit_transform(self,y):
    self.fit(y)
    return self.transform(y)
  
  def inverse_transform(self,x):
    self.check_is_fitted()
    return self.categories_[x]

class LabelBinarizer(Encoders):
  def __init__(self,neg_label=0,pos_label=1):
    self.neg_label = neg_label
    self.pos_label = pos_label

  def fit(self,y):
    super().fit(y)
    self.multiclass = len(np.unique(y)) > 2 
    if self.multiclass: self.identity_ = np.eye(len(self.categories_),dtype=int)
    return self

  def transform(self,y):
    self.check_is_fitted()
    self.check_isin(y)
    if self.multiclass:
      return self.identity_[np.searchsorted(self.categories_, y)]
    return np.searchsorted(self.categories_,y).reshape(-1,1)
  
  def fit_transform(self,y):
    self.fit(y)
    return self.transform(y)
  
  def inverse_transform(self,y):
    return self.categories_[y] if not self.multiclass else self.categories_[np.argmax(y,axis=1)]
    


  
def onehot_encoder(x): return OneHotEncoder().fit_transform(x)
def label_encoder(x): return np.unique(x),np.array([np.where(np.unique(x) == i)[0] for i in x]).reshape(-1)
def label_binarize(x) -> object: return OneHotEncoder().fit(x.reshape(-1,1)) if len(np.unique(x)) > 2 else LabelEncoder(reshape_=False).fit(x) 

if __name__ == "__main__":
  from sklearn import preprocessing
  import pandas as pd
  df = pd.read_csv("examples/dataset/drug200.csv")
  x = df['Drug'].values
  y = df['Sex'].values

  enc = preprocessing.LabelBinarizer()
  enc.fit(x)
  print(enc.transform(x[:2]))
  print(x[:2])

  enc = LabelBinarizer().fit(x)
  y_trans = enc.transform(x[:2])
  print(y_trans)
  print(enc.inverse_transform(y_trans))
  
  
