from ast import Mult
import unittest 
from minilearn.models import GaussianNB,MultinomialNB
from minilearn.utils import train_test_split
from sklearn.datasets import make_classification
from sklearn import naive_bayes
import numpy as np
from tests.models import check_attribute_model
import time


class TestGaussianNB(unittest.TestCase):
  def setUp(self) -> None:
    num_points = 100
    num_features = 5
    num_classes = 2

    x,y = make_classification(num_points,num_features,n_informative=num_features// 2, n_classes = num_classes)
    self.X_train,self.y_train,self.X_test,self.y_test = train_test_split(x,y)
  
  def test_attribute(self):
    gnb = GaussianNB().fit(self.X_train,self.y_train)
    check_attribute_model(gnb,self.X_train,self.y_train)
    gnb_ = naive_bayes.GaussianNB().fit(self.X_train,self.y_train)
    np.testing.assert_array_equal(gnb.class_prior_,gnb_.class_prior_)
    np.testing.assert_allclose(gnb.var_,gnb_.var_)
    np.testing.assert_array_equal(gnb.theta_,gnb_.theta_)
    np.testing.assert_array_equal(gnb.epsilon_,gnb_.epsilon_)

  def test_predict(self):
    gnb = GaussianNB().fit(self.X_train,self.y_train)
    gnb_ = naive_bayes.GaussianNB().fit(self.X_train,self.y_train)
    np.testing.assert_array_equal(gnb.predict(self.X_test),gnb_.predict(self.X_test))
    

class TestMultinomialNB(unittest.TestCase):
  def setUp(self) -> None:
    num_points = 100
    num_features = 5
    num_classes = 2
    x,y = make_classification(num_points,num_features,n_informative=num_features// 2, n_classes = num_classes)
    self.X_train,self.y_train,self.X_test,self.y_test = train_test_split(x,y)

  def test_attribute(self):
    X_train = np.abs(self.X_train)
    mnb = MultinomialNB().fit(X_train,self.y_train)
    mnb_ = naive_bayes.MultinomialNB().fit(X_train,self.y_train)
    check_attribute_model(mnb,X_train,self.y_train)
    np.testing.assert_allclose(mnb.class_log_prior,mnb_.class_log_prior_)
    np.testing.assert_allclose(mnb.feature_log_prob,mnb_.feature_log_prob_)

  def test_predict(self):
    X_train = np.abs(self.X_train)
    X_test = np.abs(self.X_test)

    mnb = MultinomialNB().fit(X_train,self.y_train)
    mnb_ = naive_bayes.MultinomialNB().fit(X_train,self.y_train)
    np.testing.assert_array_equal(mnb.predict(X_test),mnb_.predict(X_test))
    np.testing.assert_allclose(mnb.predict_proba(X_test),mnb_.predict_proba(X_test))
    np.testing.assert_allclose(mnb.predict_log_proba(X_test),mnb_.predict_log_proba(X_test))

  def test_performance(self):
    X_train = np.abs(self.X_train)
    X_test = np.abs(self.X_test)

    start = time.monotonic()
    mnb = MultinomialNB().fit(X_train,self.y_train)
    mnb.predict(X_test)
    mnb.predict_log_proba(X_test)
    mnb.predict_proba(X_test)
    mnb = f"times : {time.monotonic() - start}"

    start = time.monotonic()
    mnb_ = naive_bayes.MultinomialNB().fit(X_train,self.y_train)
    mnb_.predict(X_test)
    mnb_.predict_log_proba(X_test)
    mnb_.predict_proba(X_test)
    mnb_ = f"times : {time.monotonic() - start}"

    self.assertLess(mnb,mnb_)



if __name__ == "__main__":
  unittest.main()
    
