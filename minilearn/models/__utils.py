import numpy as np
from minilearn.encoders import LabelEncoder

def minhowski_distance(x1:np.ndarray,x2:np.ndarray,p:int)->np.ndarray:return np.sum(np.abs(x1 - x2) ** p, axis=1) ** (1 / p)
def sigmoid(z:np.ndarray) -> np.ndarray: return 1 / (1+np.exp(-z))
def softmax(x): return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)