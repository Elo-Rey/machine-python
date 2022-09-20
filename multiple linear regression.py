# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 23:13:52 2022

@author: Admin
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
import numpy as np
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
msk= np.random.rand(len(df))<0.8
train = cdf[msk]
test = cdf[~msk]
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (x, y)
# The coefficients
print ('Coefficients: ', regr.coef_)
y_hat= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
test_x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
print("Residual sum of squares: %.2f" % np.mean((y_hat - test_y) ** 2))
print('Variance score: %.2f' % regr.score(x, y))
# Explained variance score: 1 is perfect prediction

cdf1 = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','CO2EMISSIONS']]
msk= np.random.rand(len(df))<0.8
train1 = cdf1[msk]
test1 = cdf1[~msk]
x3 = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y3 = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (x3, y3)
# The coefficients
print ('Coefficients: ', regr.coef_)
y_hat= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
test_x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
print("Residual sum of squares: %.2f" % np.mean((y_hat - test_y) ** 2))
print('Variance score: %.2f' % regr.score(x3, y3))


















































































