import numpy as np


def minhowski_distance(x1:np.ndarray,x2:np.ndarray,p:int)->np.ndarray:return np.sum(np.abs(x1 - x2) ** p, axis=1) ** (1 / p)


  

