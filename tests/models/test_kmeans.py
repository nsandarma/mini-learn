import numpy as np
from sklearn.cluster import KMeans as SKMeans
from minilearn.models._kmeans import KMeans
from sklearn.datasets import make_blobs
import unittest


class TestKmeans(unittest.TestCase):
  def setUp(self):
    X,y = make_blobs()
    self.X = X
    self.n_cluster = len(np.unique(y))
    self.random_state = 42

  def test_attribute(self):
    model1 = KMeans(n_clusters=self.n_cluster,random_state=self.random_state)
    model1.fit(self.X)
    model2 = SKMeans(n_clusters=self.n_cluster,random_state=self.random_state).fit(self.X)
    self.assertEqual(model1.inertia_,model2.inertia_)



if __name__ == "__main__":
  unittest.main()



