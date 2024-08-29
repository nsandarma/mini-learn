import numpy as np

# for regression
def MAE(y_true:np.ndarray,y_pred:np.ndarray):return np.mean(np.abs(y_true-y_pred))
def MSE(y_true:np.ndarray,y_pred:np.ndarray):return np.mean((y_true-y_pred) ** 2)
def MAPE(y_true:np.ndarray,y_pred:np.ndarray):return np.mean(np.abs((y_true-y_pred) / y_true))
def RMSE(y_true:np.ndarray, y_pred:np.ndarray):return np.sqrt(np.sum((y_true-y_pred)**2) / len(y_true))
def r2(y_true:np.ndarray,y_pred:np.ndarray): return 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    
# for classification
def accuracy(y_true:np.ndarray,y_pred:np.ndarray):return (y_true == y_pred).mean()
def precision(y_true:np.ndarray,y_pred:np.ndarray):
  if sum(y_pred) == 0: return 0.0
  return sum((y_true[i] == 1 and y_pred[i] == 1) for i in range(len(y_true))) / sum(y_pred)
def recall(y_true:np.ndarray,y_pred:np.ndarray):pass


if __name__ == "__main__":
  x = np.random.randn(100,)
  y = np.random.randn(100,)
  print(RMSE(x,y))

  
