# -*- coding: utf-8 -*-
####

#created on '23th May 2017'
#@author : Siddhesh Dosi

####


#cross validation
#reverse engineer
#cross validation grid search (it is help to find the parameters)

import pandas as pd


train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')
#train_df1= pd.read_csv('./data/train.csv')


#features derive from existing features
#1. journy time
#2. distance
#3. Time window
#4. Demand
#5. Week and weekend


#get journy time
train_df['pickup_datetime']=pd.to_datetime(train_df['pickup_datetime'])
train_df['dropoff_datetime']=pd.to_datetime(train_df['dropoff_datetime'])
train_df['journy_time'] = train_df['dropoff_datetime']-train_df['pickup_datetime']
train_df['journy_time']=train_df['journy_time'].map(lambda x : x.seconds)

test_df['pickup_datetime']=pd.to_datetime(test_df['pickup_datetime'])
test_df['dropoff_datetime']=pd.to_datetime(test_df['dropoff_datetime'])
test_df['journy_time'] = test_df['dropoff_datetime']-test_df['pickup_datetime']
test_df['journy_time']=test_df['journy_time'].map(lambda x : x.seconds)



#convert new_user into int value
train_df.loc[train_df['new_user']=='NO','new_user']=0
train_df.loc[train_df['new_user']=='YES','new_user']=1
 
test_df.loc[test_df['new_user']=='NO','new_user']=0
test_df.loc[test_df['new_user']=='YES','new_user']=1
 
            
#derive time window using pick timestamp
# 5AM - 9AM = 0
# 10AM - 11 AM = 1
# 12PM - 15 = 2
# 16 - 21 = 3
# 22 - 04AM = 4
def get_time_window(x):
    hour = x.hour
    if hour >= 5 and hour <=9:
        return 0
    elif hour >= 10 and hour <= 11:
        return 1
    elif hour >= 12 and hour <= 15:
        return 2
    elif hour >= 16 and hour <= 21:
        return 3
    else:
        return 4
train_df['time_window']=train_df['pickup_datetime'].apply(get_time_window)

test_df['time_window']=test_df['pickup_datetime'].apply(get_time_window)


#derive day
train_df['day'] = train_df['pickup_datetime'].map(lambda x : x.weekday())

test_df['day'] = test_df['pickup_datetime'].map(lambda x : x.weekday())

#find distance using latitude and longitude
from geopy.distance import vincenty
import numpy as np
def getDistance(x):
    if (np.isnan(x['pickup_latitude']) or np.isnan(x['pickup_longitude']) or np.isnan(x['dropoff_latitude']) or np.isnan(x['dropoff_longitude'])):
        return np.nan
    else:
        pickup_location = (x['pickup_latitude'],x['pickup_longitude'])
        drop_location = (x['dropoff_latitude'],x['dropoff_longitude'])
        distance = vincenty(pickup_location,drop_location).miles
        return distance

train_df['distance'] = train_df.apply(getDistance,axis=1)

test_df['distance'] = test_df.apply(getDistance,axis=1)

#convert payment type into int
# CRD = 0
# CSH = 1
# DIS = 2
# NOC = 3
# UNK = 4

def getPaymentType(x):
    if x=='CRD':
        return 0
    elif x == 'CSH':
        return 1
    elif x == 'DIS':
        return 2
    elif x == 'NOC':
        return 3
    elif x == 'UNK':
        return 4
train_df['payment_type'] = train_df['payment_type'].apply(getPaymentType)
  
test_df['payment_type'] = test_df['payment_type'].apply(getPaymentType)    
    
columns = ['tolls_amount','tip_amount','mta_tax','surcharge','rate_code','payment_type','journy_time','day','time_window','distance','new_user','fare_amount']


#find all columns which has null value
null_value_column = train_df[columns].columns[train_df[columns].isnull().any()].tolist()

#dealing with null values for null value column
def fillNullValue(x,column_name):
    print(x[column_name])
    if (np.isnan(x[column_name])):
        print("_"*40)
        if(column_name == 'tip_account'):
            mean = tip_amount_dic[x['vendor_id']]
        else:
            mean = surcharge_dic[x['vendor_id']]
        return mean
    else:
        return x[column_name]
        
tip_amount_mean = train_df[['vendor_id','tip_amount']].groupby('vendor_id',as_index=False)['tip_amount'].mean()
surcharge_mean = train_df[['vendor_id','surcharge']].groupby('vendor_id',as_index=False)['surcharge'].mean()

tip_amount_dic = tip_amount_mean.set_index('vendor_id')['tip_amount'].to_dict()
surcharge_dic = surcharge_mean.set_index('vendor_id')['surcharge'].to_dict()

train_df['tip_amount'] = train_df[['vendor_id','tip_amount']].apply(fillNullValue,column_name='tip_amount',axis=1)
train_df['surcharge'] = train_df[['vendor_id','surcharge']].apply(fillNullValue,column_name='surcharge',axis=1)

#fill nan for new user training data set
def fillnewuser(x):
    if(np.isnan(x['new_user'])):
        print(x['new_user'])
        if(x['fare_amount']==0):
            print(x['fare_amount'])
            return 1
        else:
            return 0
    else:
        return x['new_user']
train_df['new_user']=train_df.apply(fillnewuser,axis=1)

#fill new user value for test data set
def fillnewuserFortest(x):
    if(np.isnan(x['new_user'])):
        if x['payment_type'] == 3:
            return 1
        else:
            return 0
    else:
        return x['new_user']

test_df['new_user']=test_df.apply(fillnewuserFortest,axis=1)

#change tip amount for train and test data
def change_tip_amount(x):
    if x['payment_type']==3 and x['new_user'] == 1:
        if x['tip_amount'] !=0:
            print (x['tip_amount'])
            #print (x['fare_amount'])
        return 0
    else:
        return x['tip_amount']
#change tip amount for train and test data
def change_surcharge(x):
    if np.isnan(x['surcharge']):
        if x['payment_type']==3 and x['new_user'] == 1:
            return 0.0
    else:
        return x['surcharge']
train_df['tip_amount']=train_df.apply(change_tip_amount,axis=1)
test_df['tip_amount']=test_df.apply(change_tip_amount,axis=1)
test_df['surchaege']=test_df.apply(change_surcharge,axis=1)

#fill nan value of distance
def fillDistance(x,df):
    if np.isnan(x['distance']):
        dist=(df['distance'].loc[(df['journy_time']-x['journy_time']<=300 ) & (df['journy_time']-x['journy_time'] >=-300)]).mean()
        print(dist)
        return dist
    else:
        return x['distance']
train_df['distance']=train_df[['journy_time','distance']].apply(fillDistance,df=train_df,axis=1)
test_df['distance']=test_df[['journy_time','distance']].apply(fillDistance,df=test_df,axis=1)
train_df['distance'] = train_df['distance'].fillna(train_df['distance'].mean())

#test_df['new_user']=test_df['new_user'].fillna(0)
test_df['distance'] = test_df['distance'].fillna(test_df['distance'].mean())

test_tip_amount_mean = test_df[['vendor_id','tip_amount']].groupby('vendor_id',as_index=False)['tip_amount'].mean()
test_surcharge_mean = test_df[['vendor_id','surcharge']].groupby('vendor_id',as_index=False)['surcharge'].mean()

test_tip_amount_dic = test_tip_amount_mean.set_index('vendor_id')['tip_amount'].to_dict()
test_surcharge_dic = test_surcharge_mean.set_index('vendor_id')['surcharge'].to_dict()

test_df['tip_amount'] = test_df[['vendor_id','tip_amount']].apply(fillNullValue,column_name='tip_amount',axis=1)
test_df['surcharge'] = test_df[['vendor_id','surcharge']].apply(fillNullValue,column_name='surcharge',axis=1)


#convert VID into int
def changeVID(x):
    if x == 'DST000401':
        return 0
    elif x == 'DST000481':
        return 1
    elif x == 'DST000532':
        return 2
    elif x == 'DST000543':
        return 3
    else:
        return 4
train_df['new_vendor_id']= train_df['vendor_id'].apply(changeVID)
test_df['new_vendor_id']=test_df['vendor_id'].apply(changeVID)

#take input variable and target variable
input_variable = ['tolls_amount','tip_amount','mta_tax','surcharge','rate_code','payment_type','journy_time','day','time_window','distance','new_user']
#input_variable = ['tolls_amount','mta_tax','surcharge','rate_code','payment_type','journy_time','day','time_window','distance','new_user']

target_variable=['fare_amount']


#remove all records which has new_user=1
#train_df1= train_df[train_df['new_user']!=1]

##apply distribution on distance 
import scipy.stats as stats
import pylab as pl
distance_distribution = sorted([train_df['distance']])
fit = stats.norm.pdf(distance_distribution, np.mean(distance_distribution), np.std(distance_distribution))
pl.plot(distance_distribution,fit,'-o')
pl.hist(distance_distribution,normed=True)
pl.show() 
#########create model for predict the fare amount########################

#######################################
#apply Gradient Boosting regression
#######################################
from sklearn import ensemble
gbr_final = ensemble.GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1,max_depth=9, random_state=0, loss='ls')
gbr_final.fit(train_df[input_variable],train_df[target_variable])
gbr_pred_final = gbr_final.predict(test_df[input_variable])
final_pred_df = pd.DataFrame(gbr_pred_final,columns=['fare_amount'])

final_result = pd.concat([test_df['TID'],test_df['new_user'],final_pred_df['fare_amount']],axis=1)
final_result.loc[final_result['new_user']==1,'fare_amount']=0
final_result=final_result.drop(['new_user'],axis=1)
final_result.to_csv('./output/submission.csv',index=False)
##################End the code #######################################


#######################
#parameter tunne using GRID Search for XGBOOST
#######################
from sklearn import model_selection
from sklearn import ensemble
import xgboost as xgb
gbm = xgb.XGBRegressor()

extr=  ensemble.ExtraTreesRegressor()

extr_params = {
 'n_estimators':[1000],
 'max_depth':[2,3,9],
 'max_features':[0.3,0.5],
 'random_state':[0,1]
}

xgb_params = {
'learning_rate': [0.01, 0.1],
'n_estimators': [140],
'max_depth': [2,3,5],
'gamma': [0, 1],
'subsample': [0.7, 1],
'colsample_bytree': [0.7, 1]
}
fit_params = {
'early_stopping_rounds': 30,
'eval_metric': 'mae',
'eval_set': [[train_df[input_variable],train_df[target_variable]]]
}


h=train_df.head(10000)

grid = model_selection.GridSearchCV(gbm, xgb_params, cv=3)

grid = model_selection.GridSearchCV(extr,extr_params,cv=3)
grid.fit(h[input_variable],h[target_variable])

print (grid.best_score_)
print (grid.best_params_)
print(grid.best_estimator_)
########################
#apply xgboost to improve accuracy
#######################
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
xgb_reg=XGBRegressor(learning_rate=0.1,n_estimators=2000,max_depth=9,gamma=1)
xgb_reg.fit(train_df[input_variable],train_df[target_variable])
xgb_preds=xgb_reg.predict(test_df[input_variable])
final_pred_df = pd.DataFrame(xgb_preds,columns=['fare_amount'])

final_result = pd.concat([test_df['TID'],test_df['new_user'],final_pred_df['fare_amount']],axis=1)
final_result.loc[final_result['new_user']==1,'fare_amount']=0
final_result=final_result.drop(['new_user'],axis=1)
final_result.to_csv('./output/submission.csv',index=False)


###apply extra trees regression#######
model = ensemble.ExtraTreesRegressor(n_estimators=1000, max_depth=9, max_features=0.5, n_jobs=-1, random_state=1)
model.fit(train_df[input_variable], train_df[target_variable])
extr_pred=model.predict(test_df[input_variable])
final_pred_df = pd.DataFrame(extr_pred,columns=['fare_amount'])

final_result = pd.concat([test_df['TID'],test_df['new_user'],final_pred_df['fare_amount']],axis=1)
final_result.loc[final_result['new_user']==1,'fare_amount']=0
final_result=final_result.drop(['new_user'],axis=1)
final_result.to_csv('./output/submission.csv',index=False)


###### Check Accuracy for Different regression model ##################

#save train_df 
train_df.to_csv('./output/train_df.csv')

#split training data into train data and test data
from sklearn.model_selection import train_test_split
train,test = train_test_split(train_df[columns],test_size = 0.2,random_state=42)


#find all continuous and category variable
con = ['tolls_amount','tip_amount','mta_tax','surcharge','journy_time','distance','fare_amount']
cat = ['rate_code','payment_type','day','time_window','new_user']



#save training and test data
train.to_csv('./output/train.csv',index=False)
test.to_csv('./output/test.csv',index=False)




####################################
#apply lineary regression model
####################################
from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse

Models = []
RMSE = []

Models.append('Normal Linear Regression')
reg = LinearRegression(n_jobs=-1)
reg.fit(train[input_variable],train[target_variable])
pred = reg.predict(test[input_variable])
#pred1=np.exp(pred)
Accuracy = sqrt(mse(pred,test['fare_amount']))
print('=='*20+'RMSE: '+str(Accuracy)+'=='*20)
RMSE.append(Accuracy)

#######################################
#apply Gradient Boosting regression
#######################################

Models.append('Gradient Boosting regression')
from sklearn import ensemble
gbr = ensemble.GradientBoostingRegressor()
gbr.fit(train[input_variable],train[target_variable])
gbr_pred = gbr.predict(test[input_variable])
Accuracy1  = sqrt(mse(gbr_pred,test['fare_amount']))
print('=='*20+'RMSE: '+str(Accuracy)+'=='*20)
RMSE.append(Accuracy)


#######################################
#apply LinearRegression Step2 Polynominal
#######################################
Models.append('LinearRegression Step2 Polynominal')
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
pipe = Pipeline([
('sc',StandardScaler()),
('poly',PolynomialFeatures(include_bias=True)),
('reg',LinearRegression())
])
model = GridSearchCV(pipe,param_grid={'poly__degree':[2,3,5,7]},cv=5)
model.fit(train_df[input_variable],train_df[target_variable])
degree = model.best_params_
print(degree)
pred = model.predict(test[input_variable])
Accuracy = sqrt(mse(pred,test['fare_amount']))
print('=='*20+'RMSE: '+str(Accuracy)+'=='*20)
RMSE.append(Accuracy)


from sklearn.linear_model import Lasso
pipe = Pipeline([
('sc',StandardScaler()),
('poly',PolynomialFeatures(degree=2,include_bias=True)),
('las',Lasso())
])
model = GridSearchCV(pipe,param_grid={'las__alpha':[0.0005,0.001,0.01]})
model.fit(train_df[input_variable],train_df[target_variable])
degree = model.best_params_
print(degree)
lasso=model.predict(test_df[input_variable])
final_pred_df = pd.DataFrame(lasso,columns=['fare_amount'])

final_result = pd.concat([test_df['TID'],test_df['new_user'],final_pred_df['fare_amount']],axis=1)
final_result.loc[final_result['new_user']==1,'fare_amount']=0
final_result=final_result.drop(['new_user'],axis=1)
final_result.to_csv('./output/submission.csv',index=False)

#Accuracy = sqrt(mse(pred,test['fare_amount']))


from sklearn.linear_model import ElasticNet
pipe = Pipeline([
('sc',StandardScaler()),
('poly',PolynomialFeatures(degree=2,include_bias=True)),
('en',ElasticNet())
])
model = GridSearchCV(pipe,param_grid={'en__alpha':[0.005,0.01,0.05,0.1],'en__l1_ratio':[0.1,0.4,0.8]})
model.fit(train[input_variable],train[target_variable])
degree = model.best_params_
print(degree)
elastic = model.predict(test[input_variable])
Accuracy = sqrt(mse(elastic,test['fare_amount']))


############################
#XGBoost regression
############################
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
xgb_reg=XGBRegressor(learning_rate=0.1,n_estimators=360,max_depth=3)
xgb_reg.fit(train[input_variable],train[target_variable])
xgb_preds=xgb_reg.predict(test[input_variable])
Accuracy = sqrt(mse(xgb_preds,test['fare_amount']))
#save predition result 
#test['lm_prediction_fare_amount']=pd.DataFrame(pred)
