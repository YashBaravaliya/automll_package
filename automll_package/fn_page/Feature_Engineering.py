import streamlit as st 
from function.handling_missing_value import handling_missing_value
from function.handle_categorical_variables import handling_categorical_values
from function.explore_dataset import explore_datasets 
from function.fn import *
import io

missing_value_technique=None
missing_value_summary = None

def perform_eda(data):
    st.subheader("Exploratory Data Analysis 📊")

    # Show data summary
    st.write("Data Summary:")
    st.dataframe(data.describe())

    # Show data info
    st.write("Data Info:")
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

def fe():
    with st.container():

        data = st.session_state["data"]
        cleaned_data = st.session_state["cleaned_data"]

        st.title("Feature Engineering")
        sidebar = st.selectbox("",["Explore Dataset","Handle Missing Values 🧩","Handle Categorical Variables 🧩","Scale Down-Data 📏"])
        
        if sidebar == "Handle Missing Values 🧩":
            handling_missing_value(data,cleaned_data)

        elif sidebar == "Handle Categorical Variables 🧩":
            handling_categorical_values(data,cleaned_data)

        elif sidebar == "Scale Down-Data 📏":
            handling_scaling(data,cleaned_data)
        
        elif sidebar =="Explore Dataset":
            explore_datasets(data,cleaned_data)


