import unittest 
from minilearn.models import GaussianNB,MultinomialNB
from minilearn.utils import train_test_split
from sklearn.datasets import make_classification
from sklearn import naive_bayes
import numpy as np

class TestGaussianNB(unittest.TestCase):
  def setUp(self) -> None:
    num_points = 100
    num_features = 5
    num_classes = 2

    x,y = make_classification(num_points,num_features,n_informative=num_features// 2, n_classes = num_classes)
    self.X_train,self.y_train,self.X_test,self.y_test = train_test_split(x,y)
  
  def test_attribute(self):
    gnb = GaussianNB().fit(self.X_train,self.y_train)
    gnb_ = naive_bayes.GaussianNB().fit(self.X_train,self.y_train)
    np.testing.assert_array_equal(gnb.class_prior_,gnb_.class_prior_)
    np.testing.assert_allclose(gnb.var_,gnb_.var_)
    np.testing.assert_array_equal(gnb.mean_,gnb_.theta_)
    np.testing.assert_array_equal(gnb.classes_,gnb_.classes_)
  
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
  
  def test_fit(self):
    # mnb = MultinomialNB().fit(self.X_train,self.y_train)
    mnb_ = naive_bayes.MultinomialNB().fit(self.X_train,self.y_train)
    # print(mnb.feature_log_prob)
    # print(mnb_.feature_log_prob_)



if __name__ == "__main__":
  unittest.main()
    




