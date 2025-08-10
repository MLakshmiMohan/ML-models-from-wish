import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import linearmodel as lm
import sklearn

file_path = "C:/files/kc_house_data.csv"
df = pd.read_csv(file_path)
#print(df.columns)
cdf = df[['sqft_living', 'price']]
#print(cdf)
X = cdf['sqft_living']
y = cdf['price']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

from linearmodel import myRegressor
my_model = myRegressor()
my_model.fit(X_train, y_train)
print("My model coeff",my_model.coefficient)
print("My model intercept", my_model.intercept)
my_prediction = my_model.predict(X_test)
print("R2 score of my model is: ", my_model.R2_score(my_prediction, y_test))

from sklearn.linear_model import LinearRegression
skmodel = LinearRegression()
skmodel.fit(X_train.values.reshape(-1, 1), y_train)
print("scikit-learn model coeff", skmodel.coef_)
print("scikit-learn model intercept", skmodel.intercept_)
skmodel_prediction = skmodel.predict(X_test.values.reshape(-1, 1))
print("R2 score of scikit-learn model is: ", skmodel.score(X_test.values.reshape(-1, 1), y_test))