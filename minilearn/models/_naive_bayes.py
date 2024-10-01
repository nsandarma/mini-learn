from minilearn.models._base import BaseClassifier,BaseRegressor
from minilearn.metrics import accuracy
import numpy as np
from minilearn.utils import logsumexp

__all__ = [
  "MultinominalNB",
  "GaussianNB"
]


class Bayesian(object):
  def __init__(self):
    self.is_fitted = False

  def fit(self,X,y):
    assert len(X) == len(y) , f"len(x) != len(y)"
    assert y.ndim == 1 , "reshape y"
    self.classes_ = np.unique(y)
    self.n_samples_,self.n_features_in_ = X.shape
    self.y = y

    if isinstance(self,MultinomialNB): 
      X_neg = np.min(X) > 0 
      assert X_neg, "Negative values in data passed to MultinomialNB (input X)"

    self.is_fitted = True

    return self._partial_fit(X,y)

  def predict(self,X):
    self.check_is_fitted()
    jll = self._join_log_likelihood(X)
    return self.classes_[np.argmax(jll,axis=1)]

  def predict_proba(self,X):
    self.check_is_fitted()
    return np.exp(self.predict_log_proba(X))

  def predict_log_proba(self,X):
    self.check_is_fitted()
    jll = self._join_log_likelihood(X)
    log_prob_x = logsumexp(jll,axis=1)
    return jll - np.atleast_2d(log_prob_x).T


class MultinomialNB(Bayesian,BaseClassifier):
  def __init__(self,alpha = 1.0):
    self.alpha_ = alpha

  def __str__(self):
    return "MultinomialNB"

  def _partial_fit(self,X,y):
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
    return self

  def _join_log_likelihood(self,X):
    return np.dot(X,self.feature_log_prob.T) + self.class_log_prior


class GaussianNB(Bayesian,BaseClassifier):
  def __init__(self,var_smoothing=1e-09,use_log=False):
    self.var_smoothing = var_smoothing
    self.theta_ = None
    self.var_ = None
    self.use_log = use_log

  def __str__(self):
    return "GaussianNB"

  def _partial_fit(self,X,y):
    n_samples,n_features = X.shape
    self.class_count_ = np.unique(y,return_counts=True)
    self.epsilon_ = self.var_smoothing * np.var(X, axis=0).max()

    n_classes = len(self.classes_)
    self.theta_ = np.zeros((n_classes,n_features),dtype=np.float64)
    self.var_ = np.zeros((n_classes,n_features),dtype=np.float64)
    self.class_prior_ = np.zeros(n_classes,dtype=np.float64)
    self.class_count_ = np.zeros(n_classes,dtype=np.int16)

    for idx,c in enumerate(self.classes_):
      X_c = X[y == c]
      self.theta_[idx,:] = X_c.mean(axis=0)
      self.var_[idx,:] = X_c.var(axis=0) + self.var_smoothing
      self.class_prior_[idx] = X_c.shape[0] / n_samples
      self.class_count_[idx] = len(X_c)

    self.n_features_in_ = n_features
    return self

  def _join_log_likelihood(self, X):
      joint_log_likelihood = []
      for i in range(np.size(self.classes_)):
          jointi = np.log(self.class_prior_[i])
          n_ij = -0.5 * np.sum(np.log(2.0 * np.pi * self.var_[i, :]))
          n_ij -= 0.5 * np.sum(((X - self.theta_[i, :]) ** 2) / (self.var_[i, :]), 1)
          joint_log_likelihood.append(jointi + n_ij)

      joint_log_likelihood = np.array(joint_log_likelihood).T
      return joint_log_likelihood

