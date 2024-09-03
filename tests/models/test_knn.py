import unittest
from minilearn.models import KNN,KNNRegressor,MKNN
from minilearn.datasets import load_clf,load_reg
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
import numpy as np
import random


np.random.seed(42)
class TestKNN(unittest.TestCase):
  # x_train,y_train,x_test,y_test = load_clf(n_features=10,n_samples=100,n_class=2)
  x_train,y_train,x_test,y_test = load_clf()
  k = 2
  metric = random.choice(["euclidean","manhattan"])

  def setUp(self):
    self.knn_minilearn = KNN(n_neighbors=self.k,metric=self.metric)
    self.knn_sklearn = KNeighborsClassifier(n_neighbors=self.k,metric=self.metric)

  def test_fit(self):
    knn = self.knn_minilearn.fit(self.x_train,self.y_train)
    self.assertIsInstance(knn,KNN)

  def test_pred(self):
    metric = self.metric
    sc1_,sc2_ = [],[]
    for i in range(self.k,14):
      knn1 = KNN(n_neighbors=i,metric=metric)
      knn1.fit(self.x_train,self.y_train)
      sc1 = knn1.score(self.x_test,self.y_test)
      sc1_.append(sc1)
      knn2 = KNeighborsClassifier(n_neighbors=i,metric=metric)
      knn2.fit(self.x_train,self.y_train)
      sc2 = knn2.score(self.x_test,self.y_test)
      sc2_.append(sc2)
    
    sc1,sc2 = np.mean(sc1_),np.mean(sc2_)
    diff = abs(sc2-sc1)
    print(diff)

    lim = 0.035 # lim of difference score
    self.assertLessEqual(diff,lim,"Err!")
    
    print(f"KNN MINILEARN : {np.mean(sc1_)}\nKNN SKLEARN : {np.mean(sc2_)}")
  
  def test_pred_prob(self):
    k = random.randint(2,6)
    self.knn_minilearn.fit(self.x_train,self.y_train)
    prob1 = self.knn_minilearn.predict_proba(self.x_test)
    self.knn_sklearn.fit(self.x_train,self.y_train)
    prob2 = self.knn_sklearn.predict_proba(self.x_test)
    np.testing.assert_equal(prob1,prob2)
  
  
  def test_get_params(self): 
    params = {"n_neighbors":self.k,"metric":self.metric}
    self.assertDictEqual(self.knn_minilearn.get_parameters,params)


class TestKNNRegressor(unittest.TestCase):
  # x_train,y_train,x_test,y_test = load_clf(n_features=10,n_samples=100,n_class=2)
  x_train,y_train,x_test,y_test = load_reg()
  k = 2
  metric = "euclidean"
  def test_pred(self):
    metric = self.metric
    sc1_,sc2_ = [],[]
    for i in range(self.k,14):
      knn = KNNRegressor(n_neighbors=i,metric=metric)
      knn.fit(self.x_train,self.y_train)
      sc1 = knn.score(self.x_test,self.y_test)
      sc1_.append(sc1)
      knn = KNeighborsRegressor(n_neighbors=i,metric=metric)
      knn.fit(self.x_train,self.y_train)
      sc2 = knn.score(self.x_test,self.y_test)
      sc2_.append(sc2)
    
    sc1,sc2 = np.mean(sc1_),np.mean(sc2_)
    diff = abs(sc2-sc1)
    print(diff)
    lim = 0.035 # lim of difference score
    self.assertLessEqual(diff,lim,"Err!")
    
    print(f"KNNReg MINILEARN : {np.mean(sc1_)}\nKNNReg SKLEARN : {np.mean(sc2_)}")



if __name__ == "__main__":
  unittest.main()