import numpy as np

class KMeans:
  def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, random_state=None):
      self.n_clusters = n_clusters
      self.max_iter = max_iter
      self.tol = tol
      self.random_state = random_state
      self.cluster_centers_ = None
      self.labels_ = None
      self.inertia_ = None

  def fit(self, X):
    if self.random_state:
      np.random.seed(self.random_state)
    
    # Randomly initialize centroids
    initial_indices = np.random.permutation(X.shape[0])[:self.n_clusters]
    self.cluster_centers_ = X[initial_indices]

    for i in range(self.max_iter):
      # Assign labels based on closest centroids
      self.labels_ = self._assign_labels(X)
      
      # Calculate new centroids
      new_centroids = np.array([X[self.labels_ == j].mean(axis=0) for j in range(self.n_clusters)])
      
      # Check for convergence (if centroids do not change significantly)
      if np.all(np.linalg.norm(new_centroids - self.cluster_centers_, axis=1) < self.tol):
        break
          
      self.cluster_centers_ = new_centroids
    
    self.inertia_ = self._compute_inertia(X)

  def predict(self, X):
    return self._assign_labels(X)

  def _assign_labels(self, X):
    # Calculate the distance between each point and each centroid
    distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
    return np.argmin(distances, axis=1)

  def _compute_inertia(self, X):
    # Sum of squared distances of samples to their closest cluster center
    return np.sum((X - self.cluster_centers_[self.labels_]) ** 2)



