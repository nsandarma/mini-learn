import numpy as np

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


def train_test_split(x,y,train_size=0.8,random_state=None):
  np.random.seed(random_state)
  assert len(x) == len(y), "len(x) != len(y)"
  assert train_size > 0.5 and train_size < 1.0, "train_size not in range 0.6 -> 0.9"
  n = int(train_size * len(x))
  idx = np.arange(len(x))
  np.random.shuffle(idx)
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
