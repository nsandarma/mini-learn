import numpy as np
from typing import Literal

# for regression
def MAE(y_true:np.ndarray,y_pred:np.ndarray):return np.mean(np.abs(y_true-y_pred))
def MSE(y_true:np.ndarray,y_pred:np.ndarray):return np.mean((y_true-y_pred) ** 2)
def MAPE(y_true:np.ndarray,y_pred:np.ndarray):return np.mean(np.abs((y_true-y_pred) / y_true))
def RMSE(y_true:np.ndarray, y_pred:np.ndarray):return np.sqrt(np.sum((y_true-y_pred)**2) / len(y_true))
def r2(y_true:np.ndarray,y_pred:np.ndarray):return 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    
# for classification
def accuracy(y_true:np.ndarray,y_pred:np.ndarray):return (y_true == y_pred).mean()
def confusion_matrix(y_true:np.ndarray, y_pred:np.ndarray, labels=None):
  if labels is None: labels = np.unique(np.concatenate((y_true, y_pred)))
  n_labels = len(labels)
  label_to_index = {label: i for i, label in enumerate(labels)}
  cm = np.zeros((n_labels, n_labels), dtype=int)
  for true, pred in zip(y_true, y_pred):
    true_index = label_to_index[true]
    pred_index = label_to_index[pred]
    cm[true_index, pred_index] += 1
  return cm

def precision(y_true:np.ndarray,y_pred:np.ndarray,average:Literal["binary","micro","macro","weighted"] = "binary"):
  cm = confusion_matrix(y_true,y_pred)
  if average == "binary": assert len(cm) == 2,f"Target is multiclass but average='binary',please choose average on ['micro','macro','weighted']"
  if average == "binary" :return cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0.0 # tp : cm[1,1] | fp : cm[0,1]
  elif average == "micro": return np.trace(cm) / (np.trace(cm) + np.sum(np.sum(cm,axis=0) - np.diag(cm))) if (np.trace(cm) + np.sum(np.sum(cm,axis=0) - np.diag(cm))) > 0 else 0.0
  elif average == "macro": return np.mean([cm[i,i] / (cm[i,i] + np.sum(cm[:, i]) - cm[i,i]) if (cm[i,i] + np.sum(cm[:, i]) - cm[i,i]) > 0 else 0.0 for i in range(len(cm))])
  elif average == "weighted":
    precisions = []
    supports = []
    for i in range(len(cm)):
      tp = cm[i, i]
      fp = np.sum(cm[:, i]) - tp
      precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
      support = np.sum(cm[i, :])
      precisions.append(precision * support)
      supports.append(support)
    return np.sum(precisions) / np.sum(supports) if np.sum(supports) > 0 else 0.0
  else:
    raise ValueError("Averaging method not supported: {}".format(average))

def recall(y_true:np.ndarray,y_pred:np.ndarray,average:Literal["binary","micro","macro","weighted"] = "binary"):
  cm = confusion_matrix(y_true=y_true,y_pred=y_pred)
  if average == "binary": assert len(cm) == 2,f"Target is multiclass but average='binary',please choose average on ['micro','macro','weighted']"
  if average == "binary" : return cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0.0 # tp : cm[1,1] | fn : cm[1,0]
  elif average == "micro": return np.trace(cm) / (np.trace(cm) + np.sum(np.sum(cm, axis=1) - np.diag(cm) )) if (np.trace(cm) + np.sum(np.sum(cm, axis=1) - np.diag(cm) )) > 0 else 0.0
  elif average == "macro": return np.mean([cm[i,i] / (cm[i,i] + np.sum(cm[i,:]) - cm[i,i]) if (cm[i,i] + np.sum(cm[i,:]) - cm[i,i]) > 0 else 0.0 for i in range(len(cm))])
  elif average == "weighted":
    recalls = []
    supports = []
    for i in range(len(cm)):
      tp = cm[i, i]
      fn = np.sum(cm[i, :]) - tp
      recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
      support = np.sum(cm[i, :])
      recalls.append(recall * support)
      supports.append(support)
    return np.sum(recalls) / np.sum(supports) if np.sum(supports) > 0 else 0.0
  else:
    raise ValueError("Averaging method not supported: {}".format(average))

def f1(y_true:np.ndarray,y_pred:np.ndarray,average:Literal["binary","micro","macro","weighted"] = "binary"):
  p = lambda i=None: precision(y_true==i,y_pred==i,average="binary") if average not in ["binary","micro"] else precision(y_true,y_pred,average=average)
  r = lambda i=None: recall(y_true==i,y_pred==i,average="binary") if average not in ["binary","micro"] else recall(y_true,y_pred,average=average)
  if average in ["binary","micro"]:return 2 * (p() * r()) / (p() + r()) if (p() + r()) > 0 else 0.0
  n = len(np.unique(np.concatenate((y_true, y_pred))))
  f1 = [(2 * (p(i) * r(i)) / (p(i) + r(i)) if (p(i) + r(i)) > 0 else 0.0) for i in range(n)]
  if average == 'macro': return np.mean(f1)
  support = np.array([np.sum(y_true==i) for i in range(n)])
  if average == 'weighted': return np.sum(f1*support) / np.sum(support) if np.sum(support) > 0 else 0.0
  else:
      raise ValueError("Averaging method not supported: {}".format(average))

def auc():pass
def roc_curva():pass
def roc_auc():pass


def categorical_crossentropy(y_true:np.ndarray,y_pred:np.ndarray): 
  # y_true = [[0, 1, 0], [0, 0, 1], [1, 0, 0]])
  # y_pred = [[0.8, 0.1, 0.1], [0.2, 0.3, 0.5], [0.1, 0.6, 0.3]])
  return -1/len(y_true) * np.sum(np.sum(y_true * np.log(y_pred) ))

def sparse_categorical_crossentropy(y_true,y_pred):
  y_true_onehot = np.zeros_like(y_pred)
  y_true_onehot[np.arange(len(y_true)),y_true] = 1
  return -np.mean(np.sum(y_true_onehot * np.log(y_pred),axis=-1))






if __name__ == "__main__":
  pass
