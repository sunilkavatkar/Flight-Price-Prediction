# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 17:15:35 2021

@author: sunil
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

training_dataset=pd. read_excel("Data_Train.xlsx")
training_dataset.shape
#print(training_dataset.head)

training_dataset.info()

#finding missiing vaues
training_dataset.isnull().sum()

#dropping missing values
training_dataset.dropna(inplace=True)
training_dataset.shape

#formatting date
training_dataset['journey_day']=pd.to_datetime(training_dataset['Date_of_Journey'], format="%d/%m/%Y").dt.day
training_dataset['journey_month']=pd.to_datetime(training_dataset['Date_of_Journey'], format="%d/%m/%Y").dt.month
training_dataset.drop(['Date_of_Journey'], axis=1, inplace=True)

#formatting dep time
training_dataset['hour_dep']= pd.to_datetime(training_dataset['Dep_Time']).dt.hour
training_dataset['minutes_dep']= pd.to_datetime(training_dataset['Dep_Time']).dt.minute
training_dataset.drop(['Dep_Time'], axis=1, inplace=True)

#formatting arrival time
training_dataset['hour_arrival']= pd.to_datetime(training_dataset['Arrival_Time']).dt.hour
training_dataset['minute_arrival']= pd.to_datetime(training_dataset['Arrival_Time']).dt.minute
training_dataset.drop(['Arrival_Time'], axis=1, inplace=True)

#processing Duration column
duration = list(training_dataset["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"
        else:
            duration[i] = "0h " + duration[i]           

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extract minutes
training_dataset["Duration_hours"] = duration_hours
training_dataset["Duration_mins"] = duration_mins
training_dataset.drop(["Duration"], axis=1, inplace=True)

#Encoding Airline column
training_dataset["Airline"].value_counts()
ax=sns.boxplot(x="Airline", y="Price", data=training_dataset)

# As Airline is Nominal Categorical data-(not in order)-use OneHotEncoding
Airline = training_dataset[["Airline"]]
Airline = pd.get_dummies(Airline, drop_first= True)

#Encoding Source column
training_dataset["Source"].value_counts()
ax=sns.boxplot(x="Source", y="Price", data=training_dataset)
Source = training_dataset[["Source"]]
Source = pd.get_dummies(Source, drop_first= True)

#Encoding Destination column
training_dataset["Destination"].value_counts()
ax=sns.boxplot(x="Destination", y="Price", data=training_dataset)
Destination = training_dataset[["Destination"]]
Destination = pd.get_dummies(Destination, drop_first= True)

#Dropping Route and Additional_info column as they don't have significance
training_dataset.drop(["Route", "Additional_Info"], axis=1, inplace=True)
#Encoding Total_Stops column-Ordinal Categorical-use LabelEncoder
training_dataset["Total_Stops"].value_counts()
training_dataset.replace({"non-stop":0, "1 stop":1, "2 stops":2, "3 stops":3, "4 stops":4}, inplace=True)

data_train = pd.concat([training_dataset, Airline, Source, Destination], axis = 1)
data_train.drop(["Airline", "Source", "Destination"], axis=1, inplace=True)

#Processig test data
testing_dataset=pd.read_excel("Test_set.xlsx")
testing_dataset.shape
testing_dataset.info()

#finding missiing vaues
testing_dataset.isnull().sum()

#dropping missing values
testing_dataset.dropna(inplace=True)
testing_dataset.shape

#formatting date
testing_dataset['journey_day']=pd.to_datetime(testing_dataset['Date_of_Journey'], format="%d/%m/%Y").dt.day
testing_dataset['journey_month']=pd.to_datetime(testing_dataset['Date_of_Journey'], format="%d/%m/%Y").dt.month
testing_dataset.drop(['Date_of_Journey'], axis=1, inplace=True)

#formatting dep time
testing_dataset['hour_dep']= pd.to_datetime(testing_dataset['Dep_Time']).dt.hour
testing_dataset['minutes_dep']= pd.to_datetime(testing_dataset['Dep_Time']).dt.minute
testing_dataset.drop(['Dep_Time'], axis=1, inplace=True)

#formatting arrival time
testing_dataset['hour_arrival']= pd.to_datetime(testing_dataset['Arrival_Time']).dt.hour
testing_dataset['minute_arrival']= pd.to_datetime(testing_dataset['Arrival_Time']).dt.minute
testing_dataset.drop(['Arrival_Time'], axis=1, inplace=True)

#processing Duration column
duration = list(testing_dataset["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"
        else:
            duration[i] = "0h " + duration[i]           

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extract minutes
testing_dataset["Duration_hours"] = duration_hours
testing_dataset["Duration_mins"] = duration_mins
testing_dataset.drop(["Duration"], axis=1, inplace=True)

#Encoding Airline column-Nominal Categorical data-(not in order)-use OneHotEncoding
testing_dataset["Airline"].value_counts()
Airline = testing_dataset[["Airline"]]
Airline = pd.get_dummies(Airline, drop_first= True)

#Encoding Source column
testing_dataset["Source"].value_counts()
Source = testing_dataset[["Source"]]
Source = pd.get_dummies(Source, drop_first= True)

#Encoding Destination column
testing_dataset["Destination"].value_counts()
Destination = testing_dataset[["Destination"]]
Destination = pd.get_dummies(Destination, drop_first= True)

#Dropping Route and Additional_info column as they don't have significance
testing_dataset.drop(["Route", "Additional_Info"], axis=1, inplace=True)
#Encoding Total_Stops column-Ordinal Categorical-use LabelEncoder
testing_dataset["Total_Stops"].value_counts()
testing_dataset.replace({"non-stop":0, "1 stop":1, "2 stops":2, "3 stops":3, "4 stops":4}, inplace=True)

data_test = pd.concat([testing_dataset, Airline, Source, Destination], axis = 1)
data_test.drop(["Airline", "Source", "Destination"], axis=1, inplace=True)

#model training
data_train.columns
x=data_train.loc[:, ['Total_Stops', 'journey_day', 'journey_month', 'hour_dep',
       'minutes_dep', 'hour_arrival', 'minute_arrival', 'Duration_hours',
       'Duration_mins', 'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
       'Airline_Jet Airways', 'Airline_Jet Airways Business',
       'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
       'Destination_Kolkata', 'Destination_New Delhi']]
y=data_train.iloc[:, 1]

#finding correlation
plt.figure(figsize = (16,16))
sns.heatmap(data_train.corr(), annot = True, cmap = "BrBG")
plt.show()
print(data_train.corr().index)
# Important feature using ExtraTreesRegressor

from sklearn.ensemble import ExtraTreesRegressor
selection = ExtraTreesRegressor()
selection.fit(x, y)
print(selection.feature_importances_)

plt.figure(figsize = (15,10))
feat_importances = pd.Series(selection.feature_importances_, index=x.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()
#train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

#model fitting
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()
regressor.fit(x_train, y_train)

#Prediction
y_pred=regressor.predict(x_test)
regressor.score(x_train, y_train)
regressor.score(x_test, y_test)

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2 score:', metrics.r2_score(y_test, y_pred))

#Randomized Search CV
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 50, stop = 500, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

# Create the random grid

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

regressor_opt = RandomizedSearchCV(estimator = regressor, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
regressor_opt.fit(x_train,y_train)
regressor_opt.best_params_

prediction = regressor_opt.predict(x_test)

print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))
print("R2 score",metrics.r2_score(y_test, prediction))

#saving model
import pickle
file = open('model_flight.pkl', 'wb')
pickle.dump(regressor_opt, file)

model = open('model_flight.pkl','rb')
forest = pickle.load(model)
y_prediction = forest.predict(x_test)
metrics.r2_score(y_test, y_prediction)
