# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 18:33:41 2019

@author: Gyanendra
"""

# =============================================================================
# Predicting the price of pre-owned cars
# =============================================================================

import pandas as pd
import numpy as np
#import numpy  as np
import seaborn as sns

# =============================================================================
# settinng dimensions for plot
# =============================================================================

sns.set(rc={'figure.figsize':(11.7,8.27)})

# =============================================================================
# Reading data
# =============================================================================

carsdata = pd.read_csv('cars_sampled.csv')

# =============================================================================
# Creating a copy of the data
# =============================================================================

cars=carsdata.copy()

# =============================================================================
# structure of the data
# =============================================================================
cars.info()


# =============================================================================
# Summarize the data
# =============================================================================

cars.describe()
pd.set_option('display.float_format',lambda x: '%.3f' % x)
cars.describe()

# =============================================================================
# Dropping unwanted coloumns
# =============================================================================

col = ['name','dateCrawled','dateCreated','postalCode','lastSeen']
cars=cars.drop(columns=col, axis=1)

# =============================================================================
# Removing the dublicate records
# =============================================================================

cars.drop_duplicates(keep='first',inplace=True)
#470

# =============================================================================
# Data Cleaning
# =============================================================================

#No of missing values
cars.isnull().sum()

#Variable yearsofRegistration
yearwise_count = cars['yearOfRegistration'].value_counts().sort_index()
sum(cars['yearOfRegistration']>2018)
sum(cars['yearOfRegistration']<1950)
sns.regplot(x='yearOfRegistration', y='price', scatter=True, fit_reg=False, data=cars)

#Working range 1950 and 2018

#Variable price

price_count = cars['price'].value_counts().sort_index()
sns.distplot(cars['price'])
cars['price'].describe()
sns.boxplot(y=cars['price'])
sum(cars['price']>15000)
sum(cars['price']<100)
#Working range 100 and 15000

#variable PowerPS
power_count = cars['powerPS'].value_counts().sort_index()
sns.distplot(cars['powerPS'])
cars['powerPS'].describe()
sns.boxplot(y=cars['powerPS'])
sns.regplot(x='powerPS',y='price',scatter=True,fit_reg=False,data=cars)
sum(cars['powerPS']>500)
sum(cars['powerPS']<30)
#Working range 10 and 500


# =============================================================================
# working range of data
# =============================================================================

#Working range of data

cars = cars[
            (cars.yearOfRegistration <= 2018)
           &(cars.yearOfRegistration >= 1950)
           &(cars.price >= 100)
           &(cars.price <= 15000)
           &(cars.powerPS >=10)
           &(cars.powerPS <=500)]

# -6700 records are dropped

#Further to simplify- variable reduction
# combining yearsOfRegistration and monthOfRegistratio
cars['monthOfRegistration']/=12

#Creating new variable Age by adding yearOfRegistration and monthOfRegistration
cars['Age']=(2018-cars['yearOfRegistration'])+cars['monthOfRegistration'] 
cars['Age']=round(cars['Age'],2)
cars['Age'].describe()


#Dropping yearOfRegistration and monthOfRegistration
cars=cars.drop(columns=['yearOfRegistration','monthOfRegistration'],axis=1)

#Visualizing the parameters

#Age
sns.distplot(cars['Age'])
sns.boxplot(y=cars['Age'])

#price
sns.distplot(cars['price'])
sns.boxplot(y=cars['price'])

#powerPS
sns.distplot(cars['powerPS'])
sns.boxplot(cars['powerPS'])

#Visualizing the parametrs after narrowing working range 
#Age vs price
sns.regplot(x='Age',y='price',scatter=True,fit_reg=False,data=cars)

#Cars priced higher are newer
#With increase in age, price decreases
#However some cars are priced higher with increase

#powerPS vs price

sns.regplot(x='powerPS', y='price',scatter=True, fit_reg=False, data=cars)

#Variable Seller
cars['seller'].value_counts()
pd.crosstab(cars['seller'],columns='count',normalize=True)
sns.countplot(x='offerType',data=cars)
#Fewer cars have 'commercial' => Insignificant

# Variable offerType
cars['offerType'].value_counts()
sns.countplot(x='offerType',data=cars)
# All cars have 'offer' => Insignificant

#Variable abtest
cars['abtest'].value_counts()
pd.crosstab(cars['abtest'],columns='count',normalize=True)
sns.countplot(x='abtest',data=cars)

#Equally distributed
sns.boxplot(x='abtest',y='price',data=cars)
# For every price value there is almost 50-50 distribution
#Does not effect price => Insignificant

#Variable vehicleType
cars['vehicleType'].value_counts()
pd.crosstab(cars['vehicleType'],columns='count',normalize=True)
sns.countplot(x='vehicleType',data=cars)
sns.boxplot(x='vehicleType',y='price',data=cars)
# 8 types-  limousine, small cars and station wagons max freq
# vehicleType affects price

#Variable gearbox
cars['gearbox'].value_counts()
pd.crosstab(cars['gearbox'],columns='count',normalize=True)
sns.countplot(x='gearbox',data=cars)
sns.boxplot(x='gearbox',y='price',data=cars)
#gearbox affects price

#Variable model
cars['model'].value_counts()
pd.crosstab(cars['model'],columns='count',normalize=True)
sns.countplot(x='model',data=cars)
sns.boxplot(x='model',y='price',data=cars)
#Cars are distributed over many models
#Considered in modelling

#Variable kilometer
cars['kilometer'].value_counts()
pd.crosstab(cars['kilometer'],columns='count',normalize=True)
sns.countplot(x='kilometer',data=cars)
sns.boxplot(x='kilometer',y='price',data=cars)
#considered in modelling

#Variable fuelType
cars['fuelType'].value_counts()
pd.crosstab(cars['fuelType'],columns='count',normalize=True)
sns.countplot(x='fuelType',data=cars)
sns.boxplot(x='fuelType',y='price',data=cars)
#fuelType affects price

#Variable brand
cars['brand'].value_counts()
pd.crosstab(cars['brand'],columns='count',normalize=True)
sns.countplot(x='brand',data=cars)
sns.boxplot(x='brand',y='price',data=cars)
#Cars are distributed over many brands
#Considered for modelling

#Variable notRepairedDamage
#yes - car is damaged but not rectified
#no - car was damaged but has been rectified
cars['notRepairedDamage'].value_counts()
pd.crosstab(cars['notRepairedDamage'],columns='count',normalize=True)
sns.countplot(x='notRepairedDamage',data=cars)
sns.boxplot(x='notRepairedDamage',y='price',data=cars)
#As expected, the cars that require the damage to be required
#fail under lower price ranges

# =============================================================================
# Removing insignnificant variables
# =============================================================================

col=['seller','offerType','abtest']
cars=cars.drop(columns=col,axis=1)
cars_copy=cars.copy()

# =============================================================================
# Correlation
# =============================================================================

cars_select1=cars.select_dtypes(exclude=[object])
correlation=cars_select1.corr()
round(correlation,3)
cars_select1.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]

# =============================================================================


"""
We are going to builta Linear Regression and Random Forest model
on two  sets of data
1. Data obtained by omitting rows with any missing value
2. Data obtained by imputing the missing values 
"""

# =============================================================================
# OMITTING MISSING VALUES
# =============================================================================

cars_omit=cars.dropna(axis=0)

#Coverting categorical variables to dummy variables
cars_omit=pd.get_dummies(cars_omit,drop_first=True)

# =============================================================================
# Importing Necessary Libraries
# =============================================================================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# =============================================================================
# Model Building with Ommitted data
# =============================================================================


#Separating input and output features
x1=  cars_omit.drop(['price'],axis='columns',inplace=False)
y1= cars_omit['price']

#Plotting the variable price
prices = pd.DataFrame({"1, Before":y1,"2. After":np.log(y1)})
prices.hist()

#Transforming price as a Logarithmic value
y1=np.log(y1)

#Spillting data into test and train
X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3,random_state=3)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# =============================================================================
# BaseLine Model for OMitted Data
# =============================================================================

"""
We are making a base model by using test data mean value
This is to  set a benchmark and to compare with our regression model
"""
#finding the mean for test data value
base_pred = np.mean(y_test)
print(base_pred)

#Reporting same value till length of test data 
base_root_mean_square_error = np.sqrt(mean_squared_error(y_test, base_pred))
print(base_root_mean_square_error)


# =============================================================================
# Linear Regression with omitted data
# =============================================================================

#Setting intecept as true
lgr=LinearRegression(fit_intercept=True)

#Model
model_lin1=lgr.fit(X_train,y_train)

#Predicting Model on test set
cars_predictions_lin1 = lgr.predict(X_test)

#Compute the MSE and RMSE
lin_msel = mean_squared_error(y_test,cars_predictions_lin1)
lin_rmse1 = np.sqrt(lin_msel)
print(lin_rmse1)

# R squared value
r2_lin_test1 = model_lin1.score(X_test,y_test)
r2_lin_train1 = model_lin1.score(X_train, y_train)
print(r2_lin_test1,r2_lin_train1)

#Regression diagonstics - Residual plot analysis
residuals1 = y_test-cars_predictions_lin1
sns.regplot(x=cars_predictions_lin1, y=residuals1, scatter=True, fit_reg=False)
residuals1.describe()

# =============================================================================
# Random forest Model
# =============================================================================

#Model parameters
rf = RandomForestRegressor(n_estimators =100, max_features='auto', max_depth=100,min_samples_split=10, min_samples_leaf=4, random_state=1 )

#Model
model_rf1=rf.fit(X_train,y_train)

#Predicting model on test set
cars_predictions_rf1 = rf.predict(X_test)

#Compute the MSE and RMSE
model_rf1_msel = mean_squared_error(y_test,cars_predictions_rf1)
model_rf1_rmse1 = np.sqrt(model_rf1_msel)
print(model_rf1_rmse1)

# R squared value
r2_rf_test1 = model_rf1.score(X_test,y_test)
r2_rf_train1 = model_rf1.score(X_train, y_train)
print(r2_lin_test1,r2_lin_train1)

# =============================================================================
# Model Building with Imputed Data
# =============================================================================

cars_imputed = cars.apply(lambda x:x.fillna(x.median()) if x.dtype=='float' else x.fillna(x.value_counts().index[0])) 
cars_imputed.isnull().sum()
# =============================================================================
# Model Building with Imputed data
# =============================================================================

#separating input and output features
x2= cars_imputed.drop(['price'], axis='columns', inplace=False)
y2 = cars_imputed['price']

#Plotting the variable  price
prices = pd.DataFrame({'1.Before':y2, '2. After':np.log(y2)})
prices.hist()

#Transforming price as a Logarithmic value
y2 = np.log(y2)


#Splitting  data into test and train
X_train1, X_test1, y_train1, y_test1 = train_test_split(x2, y2, test_size=0.3, random_state=3)
print(X_train1.shape, X_test1.shape, y_train1.shape, y_test1.shape)


# =============================================================================
# Baseline Model for Imputed data
# =============================================================================

"""
We are now making a base model by using test data mean value 
This is to set a benchmark and to compare with our regression model 
"""
#fining the mean for test data value
base_pred = np.mean(y_test1)
print(base_pred)

#Repeating same value till length of test data
base_pred = np.repeat(base_pred, len(y_test1))

#finding the RMSE
base_root_mean_square_error_imputed = np.sqrt(mean_squared_error(y_test1, base_pred))
print(base_root_mean_square_error_imputed)


# =============================================================================
# Linear regression with imputed data
# =============================================================================

#setting intercept as true
lgr2 = LinearRegression(fit_intercept=True)

#Model
model_lin2 = lgr2.fit(X_train1, y_train1)

#Predicting Model on test set
cars_predictions_lin2 = lgr2.predict(X_test1)

#Compute the MSE and RMSE
lin_mse2 = mean_squared_error(y_test1,cars_predictions_lin2)
lin_rmse2 = np.sqrt(lin_mse2)
print(lin_rmse2)

# R squared value
r2_lin_test2 = model_lin2.score(X_test1,y_test1)
r2_lin_train2 = model_lin2.score(X_train1, y_train1)
print(r2_lin_test2,r2_lin_train2)

# =============================================================================
# Random forest Model
# =============================================================================

#Model parameters
rf2 = RandomForestRegressor(n_estimators =100, max_features='auto', max_depth=100,min_samples_split=10, min_samples_leaf=4, random_state=1 )

#Model
model_rf2=rf2.fit(X_train1, y_train1)

#Predicting model on test set
cars_predictions_rf2 = rf2.predict(X_test1)

#Compute the MSE and RMSE
rf_mse2 = mean_squared_error(y_test1,cars_predictions_rf2)
rf_rmse2 = np.sqrt(rf_mse2)
print(rf_rmse2)

# R squared value
r2_rf_test2 = model_rf2.score(X_test1,y_test1)
r2_rf_train2 = model_rf2.score(X_train1, y_train1)
print(r2_rf_test2,r2_rf_train2)














