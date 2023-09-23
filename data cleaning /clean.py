import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRFRegressor
from sklearn import metrics

data=pd.read_csv(r'D:\BIGMARTPROJECTML\data cleaning\originaldataset\Train.csv')
# print(data)
# print(data.shape)
# data['Item_Weight'].mean()

# filling missing values in item_weight column with value
data['Item_Weight'].fillna(data['Item_Weight'].mean(),inplace=True)

data['Outlet_Size'].fillna(data['Outlet_Size'].mode()[0],inplace=True)

data.replace({'Item_Fat_Content':{'low fat':'Low Fat','LF':'Low Fat','reg':'Regular'}},inplace=True)
# print(data['Item_Fat_Content'].value_counts())

## LABLE ENCODING

encoder = LabelEncoder()
data['Item_Identifier']=encoder.fit_transform(data['Item_Identifier'])
data['Item_Fat_Content']=encoder.fit_transform(data['Item_Fat_Content'])
data['Item_Type']=encoder.fit_transform(data['Item_Type'])
data['Outlet_Identifier']=encoder.fit_transform(data['Outlet_Identifier'])
data['Outlet_Size']=encoder.fit_transform(data['Outlet_Size'])
data['Outlet_Location_Type']=encoder.fit_transform(data['Outlet_Location_Type'])
data['Outlet_Type']=encoder.fit_transform(data['Outlet_Type'])

# print(data.head())