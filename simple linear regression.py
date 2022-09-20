# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 22:11:55 2022

@author: Admin
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
df = pd.read_csv('C:/Users/Admin/Desktop/FuelConsumptionCo2.csv')
df.head()
df.info()
df.describe()
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(10)
cdf.corr()
cdf.hist()
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='black')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS,  color='green')
plt.xlabel("Cylinders")
plt.ylabel("Emission")
plt.show()
#create mask to select random rows for train test split
msk= np.random.rand(len(df))<0.8
train = cdf[msk]
test = cdf[~msk]
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)
plt.scatter( train.ENGINESIZE, train.CO2EMISSIONS, color='red')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-b')
plt.xlabel("Engine size")
plt.ylabel("Emission")
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_) )
#using fuel consumption
train_x1 = np.asanyarray(train[["FUELCONSUMPTION_COMB"]])
test_x1 = np.asanyarray(test[["FUELCONSUMPTION_COMB"]])
regr.fit(train_x1, train_y)
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)
plt.scatter( train.FUELCONSUMPTION_COMB, train.CO2EMISSIONS, color='red')
plt.plot(train_x1, regr.coef_[0][0]*train_x1 + regr.intercept_[0], '-b')
plt.xlabel("Fuel consuumption")
plt.ylabel("Emission")
test_y2_= regr.predict(test_x1)
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y2_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y2_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y2_) )
















































































