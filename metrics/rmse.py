from sklearn.metrics import mean_squared_error
import sklearn.metrics as sklm
import math

def score(y_pred):
    return str(math.sqrt(sklm.mean_squared_error(y_test, y_pred)))