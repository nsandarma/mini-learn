import numpy as np
from abc import ABC,abstractmethod

class Encoders:
  def check_is_fitted(self):
    assert self.is_fitted,f"{self.name} instance is not fitted yet. Call 'fit' with appropriate arguments before using this"

class OneHotEncoder(Encoders):
  def __init__(self,key=None):
    self.__identity = None
    self.__categories = None

  def fit(self,x:np.ndarray):
    self.__categories = np.unique(x)
    self.__identity = np.eye(self.categories_.size)
    self.__dim = x.ndim
    self.is_fitted = True

  def transform(self,x):
    self.check_is_fitted()
    x = x.reshape(-1)

    assert all(i in self.categories_ for i in x),f"value {np.unique(x)} != {self.categories_}"
    return self.identity_[[np.where(self.categories_ == k)[0][0] for k in x]]

  def fit_transform(self,x):
    self.fit(x)
    return self.transform(x)

  @property
  def categories_(self):
    return self.__categories
  
  @property
  def identity_(self):
    return self.__identity

  @property
  def key_(self):
    return self.__key

  def inverse_transform(self,x):
    self.check_is_fitted()
    res = np.array([self.categories_[np.argmax(row)] for row in x])
    return res if self.__dim == 1 else res.reshape(-1,1)
  

def onehot(x):
  return OneHotEncoder().fit_transform(x)




if __name__ == "__main__":
  x = np.arange(1,10)
  x = np.random.randint(1,10,size=(10,)).reshape(-1,1)
  print(x.ndim)
  enc = OneHotEncoder()
  enc.fit(x)
  x_enc = enc.transform(x)
  print(enc.inverse_transform(x_enc))
  print(x)
  
    
