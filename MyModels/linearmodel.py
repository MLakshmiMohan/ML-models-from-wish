import numpy as np

class myRegressor:

    def fit(self, X, y):
        #m = np.zeros(shape = y.shape)
        X = np.asarray(X)
        y = np.asarray(y)
        m = np.sum((X - X.mean())*(y - y.mean())) / np.sum((X - X.mean())**2)
        c = y.mean() - m * X.mean()
        self.coefficient = m
        self.intercept = c
        
    def predict(self, X):
        X = np.asarray(X)
        yp = self.coefficient*X + self.intercept
        return yp
    def R2_score(self, y_pred, y_true):
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        SEE = sum((y_true - y_pred)**2)
        SST = sum((y_true - np.mean(y_true))**2)
        R2score = 1 - (SEE / SST)
        return R2score