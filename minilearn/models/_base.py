from abc import ABC,abstractmethod
import inspect
import numpy as np

from minilearn.metrics import accuracy,r2
class Base(ABC):
  t = None
  is_fitted = False
  name = "Base"

  @abstractmethod
  def __init__(self): ...

  @abstractmethod
  def predict(self,x_test): ...

  @abstractmethod
  def fit(self): ...

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
  
  def check_is_fitted(self):
    assert self.is_fitted,f"{self.name} instance is not fitted yet. Call 'fit' with appropriate arguments before using this"
    return 
  
  def score(self,x_test,y_test):
    pred = self.predict(x_test)
    if self.t == "classification": res = accuracy(y_true=y_test,y_pred=pred)
    else : res = r2(y_true=y_test,y_pred=pred)
    return res
      


