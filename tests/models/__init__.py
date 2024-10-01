from minilearn.models._base import BaseClassifier,BaseRegressor,Base
import numpy as np

def check_attribute_model(model,X,y):
  n_samples,n_features = X.shape
  np.testing.assert_array_equal(model.n_samples_,n_samples)
  np.testing.assert_array_equal(model.n_features_in_,n_features)
  
  parent = model.__class__.__bases__
  if BaseClassifier in parent:
    classes,class_count = np.unique(y,return_counts=True)
    np.testing.assert_array_equal(model.classes_,classes)
    np.testing.assert_array_equal(model.class_count_,class_count)
  

