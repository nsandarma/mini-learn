import numpy as np

class MinMaxScaler:
  def __init__(self,feature_range=(0,1)):
    self.__feature_range = feature_range
    self.__min_ = None
    self.__scale_ = None
    self.__data_min_ = None
    self.__data_max_ = None
    self.__data_range_ = None

  def fit(self,X):
    X = np.asarray(X)
    self.__data_min_ = np.min(X,axis=0)
    self.__data_max_ = np.max(X,axis=0)
    self.__data_range_ = self.__data_max_ - self.__data_min_

    range_min,range_max = self.__feature_range
    self.__scale_ = (range_max - range_min) / self.__data_range_
    self.__min_ = range_min - self.__data_min_ * self.__scale_
    return self
  
  def transform(self,X) -> np.ndarray: X = np.asarray(X); return X * self.__scale_ + self.__min_
  
  def fit_transform(self,X) -> np.ndarray: return self.fit(X).transform(X)
  
  def inverse_transform(self,X) -> np.ndarray: X = np.asarray(X); return (X-self.__min_) / self.__scale_
  
  @property
  def feature_range(self) -> tuple: return self.__feature_range

  @property
  def n_features_in_(self) -> int: return len(self.__data_range_)

class StandardScaler:
  def __init__(self):
    self.__mean = None
    self.__scale = None
    self.__var = None
  
  def fit(self,X):
    X = np.asarray(X)
    self.__n_features = X.shape[1]
    self.__mean = np.mean(X,axis=0)
    self.__var = np.var(X,axis=0)
    self.__scale = np.sqrt(self.__var)
    return self
  
  def transform(self,X) -> np.ndarray: X = np.asarray(X); return (X- self.__mean) / self.__scale
  
  def fit_transform(self,X) -> np.ndarray : return self.fit(X).transform(X)

  def inverse_transform(self,X): X = np.asarray(X); return X * self.__scale + self.__mean
  
  @property
  def n_features_in_(self) -> int: return len(self.__n_features)

def minmax(X:np.ndarray,feature_range:tuple=(0,1)): return MinMaxScaler(feature_range=feature_range).fit_transform(X)

def standard(X:np.ndarray): return StandardScaler().fit_transform(X)

# !
def is_scaled(X:np.ndarray)->bool: return False if np.max(X) > 1 else True

    

  
