import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, mean_absolute_error 

df = pd.DataFrame()

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X_train=train.drop('FloodProbability',axis=1)
y_train=train['FloodProbability']

model = LinearRegression() 
  

model.fit(X_train, y_train) 
  
predictions = model.predict(test) 

df=df.assign(FloodProbability=predictions)
df=df.assign(id=test['id'])
df.to_csv('submission.csv', index=False)