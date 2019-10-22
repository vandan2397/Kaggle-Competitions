# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 14:53:35 2019

@author: Vandan
"""

import pandas as pd
import numpy as np
import seaborn as sb

df=pd.read_csv('D:\D\STUDY\Python\Kaggle Competitions\House Prediction\Prepared Dataset\Train.csv')
#print(df.head())

#no. of rows and column
df.shape

#columns having missing values
missing=df.isnull().sum()

#deleting columns which has missing values more than 50%
df.drop(['Alley'],axis=1,inplace=True)
df.drop(['PoolQC'],axis=1,inplace=True)
df.drop(['Fence'],axis=1,inplace=True)
df.drop(['MiscFeature'],axis=1,inplace=True)
df.drop(['GarageYrBlt'],axis=1,inplace=True)
df.drop(['Id'],axis=1,inplace=True)
df.drop(['SalePrice'],axis=1,inplace=True)

df['MasVnrArea'].dtypes

#filling missing values 
df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].mean())
df['BsmtQual']=df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
df['MasVnrType']=df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
df['MasVnrArea']=df['MasVnrArea'].fillna(df['MasVnrArea'].mean())
df['FireplaceQu']=df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])
df['GarageType']=df['GarageType'].fillna(df['GarageType'].mode()[0])
df['GarageCond']=df['GarageCond'].fillna(df['GarageCond'].mode()[0])
df['GarageQual']=df['GarageQual'].fillna(df['GarageQual'].mode()[0])
df['GarageCond']=df['GarageCond'].fillna(df['GarageCond'].mode()[0])
df['GarageFinish']=df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])
df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
df['BsmtFinSF1']=df['BsmtFinSF1'].fillna(df['BsmtFinSF1'].mode()[0])
df['BsmtExposure']=df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])
df['BsmtFinType2']=df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])
df['BsmtFinType1']=df['BsmtFinType1'].fillna(df['BsmtFinType1'].mode()[0])
df['Electrical']=df['Electrical'].fillna(df['Electrical'].mode()[0])

sb.heatmap(df.isnull(),yticklabels=False,cbar=False)

print(df.shape)

#combine test data with train data
test_df= pd.read_csv('D:\D\STUDY\Python\Kaggle Competitions\House Prediction\Prepared Dataset/formulatedtest.csv')

#concat train dataset with test dataset
final_df = pd.concat([df,test_df],axis=0)

#extracting categorical features
cols = df.columns
num_cols = df._get_numeric_data().columns
columns = list(set(cols)-set(num_cols))

#function for onehotcoding
def category_onehot_multcols(multcolumns):
    df_final=final_df
    i=0
    for fields in multcolumns:
        print(fields)
        df2=pd.get_dummies(final_df[fields],drop_first=True)
        
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df2.copy()
        else:
            df_final=pd.concat([df_final,df2],axis=1)
        i=i+1

    df_final=pd.concat([final_df,df_final],axis=1)
    return df_final

final_df =  category_onehot_multcols(columns)  
    
print(final_df.shape)

#deleting duplicate columns
final_df = final_df.iloc[:,~final_df.columns.duplicated()]
print(final_df.shape)            

#splitting data intpo training and testing
df_trainX = final_df.iloc[:1460,:]
df_testX = final_df.iloc[1460:,:]

salesprice = pd.read_csv('D:\D\STUDY\Python\Kaggle Competitions\House Prediction\Prepared Dataset\Train.csv')
df_trainY = salesprice['SalePrice']


#prediction using xgboost
import xgboost
classifier = xgboost.XGBRegressor()
classifier.fit(df_trainX,df_trainY)

import pickle
filename='finalized_model.pkl'
pickle.dump(classifier,open(filename,'wb'))

ypred=classifier.predict(df_testX)
print(ypred)


#storing submissions
pred = pd.DataFrame(ypred)
dataset = pd.read_csv('D:\D\STUDY\Python\Kaggle Competitions\House Prediction\Prepared Dataset\sample_submission.csv')
id1=dataset['Id'].values
dataframe = pd.concat([dataset['Id'],pred],axis=1)
dataframe.columns= ['Id','SalePrice']
dataframe.to_csv('D:\D\STUDY\Python\Kaggle Competitions\House Prediction\Prepared Dataset\sample_submission.csv')













