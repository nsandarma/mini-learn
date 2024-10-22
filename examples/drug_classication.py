from minilearn.datasets import read_csv
from minilearn.utils import train_test_split
from minilearn.models import LogisticRegression
from minilearn.scalers import MinMaxScaler
from minilearn.encoders import LabelEncoder
from minilearn.pipeline import ColumnTransformer
from sklearn import linear_model
import numpy as np



ds = read_csv("examples/dataset/drug200.csv")
ds.fit(ds.columns[:-1],ds.columns[-1])


t = [
  ("target_encoder","TargetEncoder",["Sex","BP","Cholesterol"])
]

cl = ColumnTransformer(t,remainder="passthrough").fit_from_dataset(ds)
data = MinMaxScaler().fit_transform(cl.transform(ds))
y = ds.y.reshape(-1,1)
data = np.column_stack([data,y]).astype(np.float64)
print(data.dtype)

# with open("drug200.npy","wb") as f:
#   np.save(f,data,allow_pickle=True)





