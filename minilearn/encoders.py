import numpy as np
from abc import ABC,abstractmethod

class Encoders:
  is_fitted = False

  def fit(self,x):
    self.__categories = np.unique(x)
    self.is_fitted = True
    self.__dim = x.ndim
    return self

  def check_is_fitted(self):
    assert self.is_fitted,f"{self.name} instance is not fitted yet. Call 'fit' with appropriate arguments before using this"
  
  def check_isin(self,x):
    assert all(i in self.categories_ for i in np.unique(x)),f"value {np.unique(x)} != {self.categories_}"

  @property
  def categories_(self):
    return self.__categories
  @property
  def dim(self):
    return self.__dim

class OneHotEncoder(Encoders):
  def __init__(self):
    self.__identity = None

  def fit(self,x:np.ndarray):
    super().fit(x)
    self.__identity = np.eye(self.categories_.size)
    return self

  def transform(self,x):
    self.check_is_fitted()
    self.check_isin(x)
    x = x.reshape(-1)

    return self.identity_[[np.where(self.categories_ == k)[0][0] for k in x]]

  def fit_transform(self,x):
    self.fit(x)
    return self.transform(x)
  
  @property
  def identity_(self):
    return self.__identity

  def inverse_transform(self,x):
    self.check_is_fitted()
    res = np.array([self.categories_[np.argmax(row)] for row in x])
    return res if self.dim == 1 else res.reshape(-1,1)
  
class LabelEncoder(Encoders):

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

class OrdinalEncoder(Encoders): ...
class TargetEncoder(Encoders): ...


  
def onehot(x): return OneHotEncoder().fit_transform(x)
def label(x): return LabelEncoder().fit_transform(x)

if __name__ == "__main__":
  x = np.arange(1,10)
  # x = np.random.randint(1,10,size=(10,))
  x = np.array(["a","b","c"]).reshape(-1,1)
  enc = LabelEncoder()
  enc.fit(x)
  print(LabelEncoder().fit_transform(x))
  
    
