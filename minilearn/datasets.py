from sklearn.datasets import make_classification,make_regression
import random
import numpy as np
import os


def load_clf(n_samples=None,n_features=None,n_class=None,train_size=None):
  n_samples = random.randrange(50,2000) if n_samples is None else n_samples
  n_features = random.randrange(10,100) if n_features is None else n_features
  n_class = random.randrange(2,5) if n_class is None else n_class
  train_size = random.choice(np.arange(0.6,0.9,0.1)) if train_size is None else train_size
  n_cluster = 2 if n_class == 2 else 1
  x,y = make_classification(n_samples=n_samples,n_classes=n_class,n_clusters_per_class=n_cluster,n_features=n_features)
  n = int(train_size * len(x))
  return x[:n],y[:n],x[n:],y[n:]


def load_reg(n_samples=None,n_features=None,train_size=None):
  n_samples = random.randrange(50,2000) if n_samples is None else n_samples
  n_features = random.randrange(10,100) if n_features is None else n_features
  train_size = random.choice(np.arange(0.6,0.9,0.1)) if train_size is None else train_size
  x,y = make_regression(n_samples=n_samples,n_features=n_features)
  n = int(train_size * len(x))
  return x[:n],y[:n],x[n:],y[n:]

# 2 class (binary)
def make_clf(n_samples=None,n_features=None,train_size=None,norm=True,seed=42):
  np.random.seed(seed)
  n_samples = random.randrange(50,2000) if n_samples is None else n_samples
  n_features = random.randrange(10,100) if n_features is None else n_features
  train_size = random.choice(np.arange(0.6,0.9,0.1)) if train_size is None else train_size

  x = np.random.rand(n_samples,n_features)
  if not norm : x *= 100
  mean = np.mean([np.sum(i) for i in x])
  y = np.array([0 if np.sum(i) < mean else 1 for i in x],dtype=int)
  
  n = int(train_size * len(x))
  return x[:n],y[:n],x[n:],y[n:]


def make_reg(n_samples=None,n_features=None,train_size=None,norm=True):
  n_samples = random.randrange(50,2000) if n_samples is None else n_samples
  n_features = random.randrange(10,100) if n_features is None else n_features
  train_size = random.choice(np.arange(0.6,0.9,0.1)) if train_size is None else train_size

  x = np.random.rand(n_samples,n_features)
  if not norm : x *= 100
  y = np.random.randint(1,200,size=(n_samples,))
  
  n = int(train_size * len(x))
  return x[:n],y[:n],x[n:],y[n:]


def read_csv(path,delimiter=",",return_dataset=True): 
  assert os.path.exists(path) , f"{path} not found!"
  data = np.genfromtxt(path,delimiter=delimiter,dtype=None,encoding=None,names=True)
  columns = data.dtype.names
  def convert_dtype(np_dtype):
      if np.issubdtype(np_dtype, np.integer):
          return 'int64'
      elif np.issubdtype(np_dtype, np.floating):
          return 'float64'
      else:
          return 'object'
  column_dtypes = {name: convert_dtype(data.dtype[name]) for name in columns}
  data_ =  np.empty((data.shape[0],len(columns)),dtype="object")
    
  for idx,col in enumerate(columns):
    data_[:,idx] = data[col]
  if return_dataset: return Dataset(data=data_,column_names=columns,column_dtypes=column_dtypes) 
  return data_,columns

class Dataset():
  def __str__(self):
    return f"<Dataset>"

  def __init__(self,data:np.ndarray,column_names:tuple,column_dtypes=None):
    self.__data = data
    self.__columns = column_names
    self.__dtypes = column_dtypes
    self.n_samples,self.n_columns = data.shape
  
  def fit(self,features,target):
    assert set(features) <= set(self.columns), f"{features} != {self.columns}"
    assert target in self.columns , f"target : {target} not in {self.columns}"
    target_col = self.columns.index(target)
    feature_col = [self.columns.index(col) for col in features]
    self.X = self.data[:,feature_col]
    self.y = self.data[:,target_col]
  
  @property
  def data(self):
    return self.__data
  
  @property
  def columns(self):
    return self.__columns

  @property
  def dtype(self):
    return self.__data.dtype

  @property
  def dtypes(self):
    return self.__dtypes


if __name__ == "__main__":
  df = read_csv("examples/dataset/drug200.csv",return_dataset=True)
  features = ["Age","Cholesterol"]
  df.fit(features,"Drug")
  print(df.data)

  
    