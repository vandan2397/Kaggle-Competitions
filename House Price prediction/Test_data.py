# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 14:54:28 2019

@author: Vandan
"""

import pandas as pd
import numpy as np
import seaborn as sb

df1=pd.read_csv('D:\D\STUDY\Python\Kaggle Competitions\House Prediction\Prepared Dataset/Test.csv')
print(df1.head())

df1.shape

#columns having missing values
missing2 = df1.isnull().sum()

#deleting columns which has missing values more than 50%
df1.drop(['Alley'],axis=1,inplace=True)
df1.drop(['PoolQC'],axis=1,inplace=True)
df1.drop(['Fence'],axis=1,inplace=True)
df1.drop(['MiscFeature'],axis=1,inplace=True)
df1.drop(['Id'],axis=1,inplace=True)
df1.drop(['GarageYrBlt'],axis=1,inplace=True)

#filling missing values
df1['LotFrontage']=df1['LotFrontage'].fillna(df1['LotFrontage'].mean())
df1['MSZoning']=df1['MSZoning'].fillna(df1['MSZoning'].mode()[0])
df1['Utilities']=df1['Utilities'].fillna(df1['Utilities'].mode()[0])
df1['Exterior1st']=df1['Exterior1st'].fillna(df1['Exterior1st'].mode()[0])
df1['Exterior2nd']=df1['Exterior2nd'].fillna(df1['Exterior2nd'].mode()[0])
df1['MasVnrType']=df1['MasVnrType'].fillna(df1['MasVnrType'].mode()[0])
df1['MasVnrArea']=df1['MasVnrArea'].fillna(df1['MasVnrArea'].mean())
df1['BsmtQual']=df1['BsmtQual'].fillna(df1['BsmtQual'].mode()[0])
df1['BsmtFullBath']=df1['BsmtFullBath'].fillna(df1['BsmtFullBath'].mode()[0])
df1['BsmtHalfBath']=df1['BsmtHalfBath'].fillna(df1['BsmtHalfBath'].mode()[0])
df1['KitchenQual']=df1['KitchenQual'].fillna(df1['KitchenQual'].mode()[0])
df1['FireplaceQu']=df1['FireplaceQu'].fillna(df1['FireplaceQu'].mode()[0])
df1['GarageType']=df1['GarageType'].fillna(df1['GarageType'].mode()[0])
df1['GarageCond']=df1['GarageCond'].fillna(df1['GarageCond'].mode()[0])
df1['GarageQual']=df1['GarageQual'].fillna(df1['GarageQual'].mode()[0])
df1['GarageFinish']=df1['GarageFinish'].fillna(df1['GarageFinish'].mode()[0])
df1['Functional']=df1['Functional'].fillna(df1['Functional'].mode()[0])
df1['GarageYrBlt']=df1['GarageYrBlt'].fillna(df1['GarageYrBlt'].mean())
df1['GarageCars']=df1['GarageCars'].fillna(df1['GarageCars'].mean())
df1['GarageArea']=df1['GarageArea'].fillna(df1['GarageArea'].mean())
df1['SaleType']=df1['SaleType'].fillna(df1['SaleType'].mode()[0])
df1['BsmtExposure']=df1['BsmtExposure'].fillna(df1['BsmtExposure'].mode()[0])
df1['BsmtCond']=df1['BsmtCond'].fillna(df1['BsmtCond'].mode()[0])
df1['BsmtFinType2']=df1['BsmtFinType2'].fillna(df1['BsmtFinType2'].mode()[0])
df1['BsmtFinType1']=df1['BsmtFinType1'].fillna(df1['BsmtFinType1'].mode()[0])
df1['BsmtFinSF1']=df1['BsmtFinSF1'].fillna(df1['BsmtFinSF1'].mode()[0])
df1['BsmtFinSF2']=df1['BsmtFinSF2'].fillna(df1['BsmtFinSF2'].mode()[0])
df1['BsmtUnfSF']=df1['BsmtUnfSF'].fillna(df1['BsmtUnfSF'].mode()[0])
df1['TotalBsmtSF']=df1['TotalBsmtSF'].fillna(df1['TotalBsmtSF'].mode()[0])

sb.heatmap(df1.isnull(),yticklabels=False,cbar=False)

print(df1['BsmtUnfSF'].value_counts())
print(df1['TotalBsmtSF'].describe())
print(df1.shape)

#copying test data to formulatedtest file
df1.to_csv('D:\D\STUDY\Python\Kaggle Competitions\House Prediction\Prepared Dataset/formulatedtest.csv',index=False)










