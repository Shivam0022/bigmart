import streamlit as st
import pandas as pd
import sklearn
from xgboost import XGBRFRegressor

@st.cache_data
def reading_csv(main_dataset):
    return pd.read_csv(main_dataset)

@st.cache_data
def train_dataset(main_dataset):
    X = main_dataset.drop(columns='Item_Outlet_Sales',axis=1)
    Y = main_dataset['Item_Outlet_Sales']
    return X,Y


t1,t2  = st.tabs(["Info","Predict"])
# Reading Dataset
main_dataset = reading_csv(r"dataset/clean_data.csv")
column_parameter,target = train_dataset(main_dataset)

# fitting model
regressor=XGBRFRegressor()
regressor.fit(column_parameter,target)
st.dataframe(column_parameter)
# Sidebar
st.sidebar.title("Parameters")
item_identifier_value = st.sidebar.selectbox("Item Identifier",list(range(0,1558)))
Item_Weight_value = st.sidebar.slider("Item Weight",4.55,21.35)
Item_Fat_Content_value = st.sidebar.radio("Item Fat Content",("Low Fat","Regular Fat"))
if Item_Fat_Content_value == "Low Fat":
    Item_Fat_Content_value_main = int(0)
if Item_Fat_Content_value == "Regular Fat":
    Item_Fat_Content_value_main = int(1)
Item_Visibility_value = st.sidebar.slider("Item Visibility",0.0000,0.3284)
Item_type_value = st.sidebar.selectbox("Item type",list(range(0,16)))
Item_MRP_value = st.sidebar.slider("Item MRP",31.29,266.88)
Outlet_Identifier_value = st.sidebar.selectbox("Outlet Identifier",list(range(0,10)))
Outlet_Establishment_Year_value = st.sidebar.selectbox("Outlet Establishment Year",list(range(1985,2010)))
Outlet_Size_value = st.sidebar.slider("Outlet Size",0,2)
Outlet_Location_type_value = st.sidebar.slider("Outlet Location type",0,2)
Outlet_type_value = st.sidebar.slider("Outlet type",0,3)
Item_Outlet_Sales_value = st.sidebar.slider("Item Outlet Sales",33.29,13086.96)