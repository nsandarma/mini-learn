import numpy as np
from numpy import dtype
import os

def read_csv(path, delimiter=",", return_dataset=True):
  # Memastikan file ada
  assert os.path.exists(path), f"{path} not found!"
  
  # Membaca CSV menggunakan genfromtxt, mengisi nilai kosong dengan np.nan
  data = np.genfromtxt(path,
   delimiter=delimiter, 
   dtype = None,
   encoding=None, 
   names=True, 
   missing_values='', 
   usemask=False,
   filling_values=0,
  )
  
  columns = data.dtype.names

  column_dtypes = {name: ('int64' if np.issubdtype(data.dtype[name], np.integer) else
                          'float64' if np.issubdtype(data.dtype[name], np.floating) else
                          'object')
                   for name in columns}

  data_ = np.array([tuple(row) for row in data], dtype='object')
  data_[data_ == ''] = np.nan
  
  if return_dataset:
    return Dataset(data=data_, column_names=columns, column_dtypes=column_dtypes)

  return data_, columns

class Dataset():
  def __str__(self):
    n_rows,n_cols = self.shape
    return f"<Dataset rows : {n_rows} | columns : {n_cols}>"

  def __init__(self,data:np.ndarray,column_names:tuple,column_dtypes=None):
    self.__data = data
    self.__columns = column_names
    self.__dtypes = column_dtypes
    self.n_samples,self.n_columns = data.shape
    self.is_fitted = False

  def __getitem__(self,indexer):
    if isinstance(indexer,(np.ndarray,list,tuple,set)): 
      indexer = [self.columns.index(i) for i in indexer]
      return self.data[:,indexer]
    else: 
      if isinstance(indexer,str):
        idx = self.columns.index(indexer)
        return self.data[:,idx].astype(self.dtypes[indexer])
      return self.data[indexer]
  
  def fit(self,features,target):
    assert set(features) <= set(self.columns), f"{features} != {self.columns}"
    assert target in self.columns , f"target : {target} not in {self.columns}"
    target_col = self.columns.index(target)
    feature_col = [self.columns.index(col) for col in features]
    self.X = self.data[:,feature_col]
    self.y = self.data[:,target_col]
    self.X_feature_names = np.array(self.columns)[feature_col]
    self.y_target_names = np.array(self.columns)[target_col]
    self.is_fitted = True
    return self
  
  def head(self,samples=5):
    samples = abs(samples)
    if samples > self.n_samples: samples = self.n_samples
    return self[:samples]

  def tail(self,samples=5):
    if samples > 0 : samples = -samples
    if samples < -self.n_samples: samples = -self.n_samples
    return self[samples:]

  def save(self,filename):
    with open(filename,"wb") as f:
      np.save(f,self.data)
  
  def save_obj(self,filename):
    pass
  
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
  
  @property
  def shape(self):
    return self.data.shape
  

# generate_synthetic_data

def make_classification(n_samples=100,n_features=2,n_classes=2,mean_range=(0,5),std=1.0,random_state=None):
  if random_state is not None:
    np.random.seed(random_state)
  samples_per_class = n_samples // n_classes
  X = []
  y = []
  
  for class_label in range(n_classes):
    mean = np.random.uniform(mean_range[0], mean_range[1], n_features)
    X_class = np.random.normal(loc=mean, scale=std, size=(samples_per_class, n_features))
    y_label = np.full(samples_per_class,class_label)
    X.append(X_class)
    y.append(y_label)
  X = np.vstack(X)
  y = np.hstack(y)
  indices = np.random.permutation(n_samples)
  return X[indices],y[indices]


if __name__ == "__main__":
  X,y = make_classification(n_features=5,n_classes=2,random_state=42)

