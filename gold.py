import pandas as pd 
import numpy as np
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

gold_data = pd.read_csv('gold.csv')
#print(gold_data.head)
#print(gold_data.isnull().sum())
#print(gold_data.describe())

correlation=gold_data.corr()
plt.figure(figsize=(8,8))

#sns.heatmap(correlation, cbar=True, square=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap='Blues')
#plt.show()

#print(correlation['GLD'])
#sns.histplot(gold_data['GLD'],color='green')
X=gold_data.drop(['Date','GLD'],axis=1)
#print(X)
Y=gold_data['GLD']
#print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=2)
regressor=RandomForestRegressor(n_estimators=100)
regressor.fit(X_train,Y_train)
test_data_prediction = regressor.predict(X_test)
#print(test_data_prediction)
error_score = metrics.r2_score(Y_test, test_data_prediction)
#print(error_score)
Y_test=list(Y_test)
plt.plot(Y_test, color='blue', label = 'Actual Value')
plt.plot(test_data_prediction, color='green', label='Predicted Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()

st.title('Gold price prediction')
st.write(gold_data)