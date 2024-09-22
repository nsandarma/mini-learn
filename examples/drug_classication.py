import pandas as pd
from minilearn.models import KNN,LogisticRegression
from sklearn import naive_bayes
from minilearn.encoders import OneHotEncoder,LabelBinarizer,LabelEncoder,TargetEncoder,OrdinalEncoder
from minilearn.utils import train_test_split
from minilearn.metrics import accuracy
import numpy as np

from minilearn.models._naive_bayes import NaiveBayes

df = pd.read_csv("examples/dataset/drug200.csv")


features = ["Sex","Cholesterol","BP","Na_to_K","Age"]
target = "Drug"

X = df[features].values
y = df[target].values



x_cat = X[:,0:3]

enc = OrdinalEncoder().fit(x_cat)
x_cat_transform = enc.transform(x_cat)


X[:,0:3] = x_cat_transform
enc = LabelEncoder().fit(y)
y = enc.transform(y)


x_train,y_train,x_test,y_test = train_test_split(X,y)
print(np.stda())

# nb = NaiveBayes()
# nb.fit(x_train,y_train)
# pred = nb.predict(x_test)
# print(accuracy(y_test,pred))

# nb = naive_bayes.GaussianNB()
# nb.fit(x_train,y_train)
# pred = nb.predict(x_test)
# print(accuracy(y_test,pred))






