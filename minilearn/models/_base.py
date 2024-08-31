from abc import ABC,abstractmethod
import inspect
from minilearn.metrics import accuracy,r2
import numpy as np

class Base(ABC):
  t = None
  is_fitted = False
  name = "Base"

  @abstractmethod
  def __init__(self): ...

  @abstractmethod
  def fit(self): ...

  @abstractmethod
  def predict(self,X): ...

  @abstractmethod 
  def score(self,X,y):...

  def fit_predict(self,X_train:np.ndarray,y_train:np.ndarray,X_test:np.ndarray):
    self.fit(X_train,y_train)
    return self.predict(X_test)

  def set_params(self,**params):
    if not params:
      return self
    valid_params = self.get_parameters
    assert all(k in valid_params.keys() for k in params.keys()), f"params is not in{valid_params.keys()}"
    [setattr(self,key,value) for key,value in params.items()]
    return self

  @property
  def get_parameters(self) -> dict:
    init_signature =  inspect.signature(self.__init__)
    params = [name for name in init_signature.parameters if name != 'self']
    return {p:getattr(self,p) for p in params}
  
  @property
  def n_features_in_(self) -> int:
    return self.X.shape[1]
  
  def check_is_fitted(self):
    assert self.is_fitted,f"{self.name} instance is not fitted yet. Call 'fit' with appropriate arguments before using this"
    return 

      
class BaseClassifier(Base):
  def score(self,X,y):return accuracy(y_true=y,y_pred=self.predict(X))

  @property
  def classes_(self): self.check_is_fitted(); return np.unique(self.y)

class BaseRegressor(Base):

  def score(self,X,y):return r2(y_true=y,y_pred=self.predict(X))