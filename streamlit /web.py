import streamlit as st
import pandas as pd

@st.clean_data
def reading_csv(dataset):
    return pd.read_csv(dataset)

# Reading Dataset
main_dataset = reading_csv("dataset/clean_data.clean_data.csv")

st.dataframe(main_dataset)

