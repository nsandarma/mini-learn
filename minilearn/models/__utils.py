import numpy as np
from minilearn.encoders import LabelEncoder

def minhowski_distance(x1:np.ndarray,x2:np.ndarray,p:int)->np.ndarray:return np.sum(np.abs(x1 - x2) ** p, axis=1) ** (1 / p)
def sigmoid(z:np.ndarray) -> np.ndarray: return 1 / (1+np.exp(-z))
def softmax(x): return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


def type_of_target(y):
  y = np.asarray(y)
  assert y.ndim == 1, "reshape !"
  assert len(y) > 1 , "!"
  if y.dtype.kind in ["U","O"]: return "binary" if len(np.unique(y)) == 2 else "multiclass"
  if not np.count_nonzero(y - np.floor(y)): return "binary" if len(y) == 2 else "multiclass"
  return "continuous"



