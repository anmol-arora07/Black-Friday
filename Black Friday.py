# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 11:18:14 2018

@author: Anmol.K
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
num=LabelEncoder()
df_test=pd.read_csv("test.csv")
df_train=pd.read_csv("train.csv")
user=df_test[['User_ID','Product_ID']]
for i in df_train,df_test:
    i['Sex']=num.fit_transform(i['Gender'])
    i['City']=num.fit_transform(i['City_Category'])
    i['Stay_In_Current_City_Years'].replace('4+',4,inplace=True)
    i['Age Category']=num.fit_transform(i['Age'])
    i['Stay']=i['Stay_In_Current_City_Years'].astype('int')
    i['User']=num.fit_transform(i['User_ID'])
    i['Product']=num.fit_transform(i['Product_ID'])
    i.drop(['User_ID','Product_ID','Gender','Age','City_Category','Stay_In_Current_City_Years'],
           axis=1,inplace=True)
    
label=df_train['Purchase']
features=df_train.drop(['Purchase'],axis=1)
features.fillna(0,inplace=True)
df_train.fillna(0,inplace=True)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features,label, test_size=0.30, random_state=101)
import xgboost as xgb
model=xgb.XGBRegressor(max_depth=8,learning_rate=0.08,n_estimators=1500,colsample_bylevel=0.9,
                       colsample_bytree=0.8,silent=False,n_jobs=-1,)
print("Implementing model..")
model.fit(X_train,y_train)
predictions=model.predict(X_test)
from math import sqrt
from sklearn.metrics import mean_squared_error
test_error=sqrt(mean_squared_error(y_test,predictions))
print(test_error)
predict=model.predict(df_test)
predict=pd.Series(predict,name="Purchase")
submission=pd.concat([user,predict],axis=1)
submission.to_csv("Submission.csv")