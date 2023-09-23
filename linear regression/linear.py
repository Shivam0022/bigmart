

#  SPLIT FEATURE AND TARGET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRFRegressor
from sklearn import metrics
import joblib
data = pd.read_csv(r'D:\BIGMARTPROJECTML\data leaning\cleaning dataset\clean_data.csv')
X=data.drop(columns='Item_Outlet_Sales',axis=1)
Y=data['Item_Outlet_Sales']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.2, random_state=2)
# print(X.shape,X_train.shape,Y.shape,Y_train.shape,X_test.shape,Y_test.shape)
# print(Y_train)



from sklearn.linear_model import LinearRegression
lr= LinearRegression()


lr.fit(X_train,Y_train)

# model save

filename = 'LINEAR_X.sav'
joblib.dump(lr,filename )

loaded_model = joblib.load(filename)
result = loaded_model.score(X_test, Y_test)
print(result)