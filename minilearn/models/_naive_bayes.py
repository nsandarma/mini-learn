from minilearn.models._base import BaseClassifier,BaseRegressor
from minilearn.metrics import accuracy
import numpy as np

__all__ = [
  "MultinominalNB",
  "GaussianNB"
]

class Bayesian:
  is_fitted = False
  def __init__(self): ...

  def fit(self,X,y):
    assert len(X) == len(y) , f"len(x) != len(y)"
    assert y.ndim == 1 , "reshape y"
    self.is_fitted = True
  
  def check_is_fitted(self):
    assert self.is_fitted , ""


class MultinomialNB():
  def __init__(self,alpha = 1.0):
    self.alpha_ = alpha

  def fit(self,X,y):
    b,n = X.shape
    self.classes_ = np.unique(y)
    n_classes = len(self.classes_)
    n_samples,n_features = X.shape

    self.class_count_ = np.zeros(n_classes,dtype=np.float64)
    self.feature_count_ = np.zeros((n_classes,n_features),dtype=np.float64)

    for idx,cls in enumerate(self.classes_):
      X_c = X[y == cls]
      self.class_count_[idx] = X_c.shape[0]
      self.feature_count_[idx,:] = X_c.sum(axis=0)
    
    self.class_log_prior = np.log(self.class_count_ / n_samples)
    smoothed_fc  = self.feature_count_ + self.alpha_
    smoothed_cc = smoothed_fc.sum(axis=1,keepdims=True)
    self.feature_log_prob = np.log(smoothed_fc) - np.log(smoothed_cc)
  
  def predict(self,X): ...


class GaussianNB():
  def __init__(self,var_smoothing=1e-09,use_log=False):
    self.var_smoothing = var_smoothing
    self.classes_ = None
    self.mean_ = None
    self.var_ = None
    self.use_log = use_log

  def fit(self,X,y):
    n_samples,n_features = X.shape
    self.classes_ = np.unique(y)
    n_classes = len(self.classes_)
    self.mean_ = np.zeros((n_classes,n_features),dtype=np.float64)
    self.var_ = np.zeros((n_classes,n_features),dtype=np.float64)
    self.class_prior_ = np.zeros(n_classes,dtype=np.float64)
    self.count_ = np.zeros(n_classes)

    for idx,c in enumerate(self.classes_):
      X_c = X[y == c]
      self.mean_[idx,:] = X_c.mean(axis=0)
      self.var_[idx,:] = X_c.var(axis=0) + self.var_smoothing
      self.class_prior_[idx] = X_c.shape[0] / n_samples
      self.count_[idx] = len(X_c)

    self.n_features_in_ = n_features
    return self

  def _gaussian_probability(self,class_idx,x):
    mean = self.mean_[class_idx]
    var = self.var_[class_idx]
    numerator = np.exp(- (x - mean) ** 2 / (2 * var))
    denominator = np.sqrt(2 * np.pi * var)
    return numerator / denominator
  
  def _predict_log_proba(self, X):
    log_proba = np.zeros((X.shape[0], len(self.classes_)))
    for i in range(X.shape[0]):
      for idx, cls in enumerate(self.classes_):
        prior = np.log(self.class_prior_[idx] + self.var_smoothing)
        likelihood = np.sum(np.log(self._gaussian_probability(idx, X[i]) + self.var_smoothing))  # Tambahkan epsilon ke probabilitas
        log_proba[i, idx] = prior + likelihood
    return log_proba

  def predict(self, X):
    log_proba = self._predict_log_proba(X)
    return self.classes_[np.argmax(log_proba, axis=1)]

  def predict_proba(self, X):
    log_proba = self._predict_log_proba(X)
    proba = np.exp(log_proba)
    return proba / np.sum(proba, axis=1, keepdims=True)
  



class NaiveBayes:
  def __init__(self,metric="gaussian",use_log=False):
    self.metric = metric
    self.use_log = use_log

  def fit(self,x,y):
    self.x = np.asarray(x)
    self.y = np.asarray(y)
    self.data = np.column_stack((x,y))
    self.n_features = self.x.shape[1]
    self.classes = set(y)
    self.n_classes = len(self.classes)
    self.summary = {}
    for i in self.classes : 
      d = self.data[self.data[:,-1] == i]
      mean = np.mean(d,axis=0)[:-1]
      std = np.std(d,axis=0)[:-1]
      l = len(d)
      self.summary[i] = dict(mean=mean,std=std,l=l)

  def get_prob(self,inp,mean,std):
    if self.metric == "gaussian":
      exponent = np.exp(-(((inp-mean)**2)/(2*(std**2))))
      res = (1 / (np.sqrt(2 * np.pi) * std)) * exponent
      if self.use_log : 
        return np.log(1+res)
      return res

  def predict(self,x):
    results = []
    for inp in x :
      pred_class = -1
      pred_prob = 0
      for i in self.classes:
        summary = self.summary[i]
        probs = self.get_prob(inp,summary["mean"],summary["std"])
        class_prob = np.prod(probs)
        if class_prob > pred_prob :
          pred_class = i
          pred_prob = class_prob
      results.append(pred_class)
    return results

if __name__ == "__main__":
  from sklearn.datasets import make_classification
  from minilearn.utils import train_test_split
  from sklearn import naive_bayes
  import numpy as np

  np.random.seed(42)

  num_features = 10
  num_classes = 6
  num_points = 20
  metric = "gaussian"

  x,y = make_classification(num_points,num_features,n_informative=num_features// 2, n_classes = num_classes)
  X_train,y_train,X_test,y_test = train_test_split(x,y)

  # nb = NaiveBayes()
  # nb.fit(X_train,y_train)
  # pred = nb.predict(X_test)
  # print(pred)

  nb = GaussianNB()
  nb.fit(X_train,y_train)
  pred = nb.predict_proba(X_test)
  print(pred)

  nb = naive_bayes.GaussianNB()
  nb.fit(X_train,y_train)
  pred = nb.predict_proba(X_test)
  print(pred)


