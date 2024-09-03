from sklearn.datasets import make_classification,make_regression
import random
import numpy as np


def load_clf(n_samples=None,n_features=None,n_class=None,train_size=None):
  n_samples = random.randrange(50,2000) if n_samples is None else n_samples
  n_features = random.randrange(10,100) if n_features is None else n_features
  n_class = random.randrange(2,5) if n_class is None else n_class
  train_size = random.choice(np.arange(0.6,0.9,0.1))
  n_cluster = 2 if n_class == 2 else 1
  x,y = make_classification(n_samples=n_samples,n_classes=n_class,n_clusters_per_class=n_cluster,n_features=n_features)
  n = int(train_size * len(x))
  return x[:n],y[:n],x[n:],y[n:]


def load_reg(n_samples=None,n_features=None,train_size=None):
  n_samples = random.randrange(50,2000) if n_samples is None else n_samples
  n_features = random.randrange(10,100) if n_features is None else n_features
  train_size = random.choice(np.arange(0.6,0.9,0.1))
  x,y = make_regression(n_samples=n_samples,n_features=n_features)
  n = int(train_size * len(x))
  return x[:n],y[:n],x[n:],y[n:]

# 2 class (binary)
def make_clf(n_samples=None,n_features=None,train_size=None,norm=True):
  n_samples = random.randrange(50,2000) if n_samples is None else n_samples
  n_features = random.randrange(10,100) if n_features is None else n_features
  train_size = random.choice(np.arange(0.6,0.9,0.1))

  x = np.random.rand(n_samples,n_features)
  if not norm : x *= 100
  mean = np.mean([np.sum(i) for i in x])
  y = np.array([0 if np.sum(i) < mean else 1 for i in x])
  
  n = int(train_size * len(x))
  return x[:n],y[:n],x[n:],y[n:]


def make_reg(n_samples=None,n_features=None,train_size=None,norm=True):
  n_samples = random.randrange(50,2000) if n_samples is None else n_samples
  n_features = random.randrange(10,100) if n_features is None else n_features
  train_size = random.choice(np.arange(0.6,0.9,0.1))

  x = np.random.rand(n_samples,n_features)
  if not norm : x *= 100
  y = np.random.randint(1,200,size=(n_samples,))
  # y = np.array([np.sum(i)**2 for i in x])
  
  n = int(train_size * len(x))
  return x[:n],y[:n],x[n:],y[n:]
