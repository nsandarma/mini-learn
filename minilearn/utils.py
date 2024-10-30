import numpy as np
from minilearn.datasets import Dataset

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


def train_test_split(x,y,test_size=None,train_size=None,random_state=None,shuffle=True):
  assert type(x) == type(y), "type difference between x and y"
  assert len(x) == len(y), "len(x) != len(y)"

  n_ = len(x)
  n = 0
  if test_size:
    assert isinstance(test_size,(int,float)), "test_size must float/int"
    if isinstance(test_size,int):
      train_size = n_ - test_size
    else:
      train_size = n_ - (test_size * n_)

  elif train_size:
    assert isinstance(train_size,(int,float)), "train_size must float/int"
    if isinstance(train_size,float):
      train_size = train_size * n_
  
  assert train_size > 0 and train_size < n_, "index out of range"

  np.random.seed(random_state)
  n = int(train_size)
  idx = np.arange(len(x))
  if shuffle: np.random.shuffle(idx)
  if isinstance(x,Dataset):
    x_train,y_train = x.get(idx[:n]),y.get(idx[:n])
    x_test,y_test = x.get(idx[n:]),y.get(idx[n:])
  else:
    x_train,y_train = x[idx[:n]],y[idx[:n]]
    x_test,y_test = x[idx[n:]],y[idx[n:]]
  return x_train,y_train,x_test,y_test

def logsumexp(x, axis=None, keepdims=False):
    """
    Compute the log of the sum of exponentials of input elements along a given axis.
    
    Parameters:
    - x: Input array
    - axis: Axis along which to compute the logsumexp. If None, computes over the entire array.
    - keepdims: If True, retains reduced dimensions with length 1.
    
    Returns:
    - result: The logsumexp value along the specified axis.
    """
    xmax = np.max(x, axis=axis, keepdims=True)
    
    exp_diff = np.exp(x - xmax)
    sum_exp = np.sum(exp_diff, axis=axis, keepdims=keepdims)
    
    logsumexp_result = np.log(sum_exp) + np.squeeze(xmax, axis=axis)
    
    if keepdims:
        return logsumexp_result
    else:
        return np.squeeze(logsumexp_result)
