import streamlit as st 
import pandas as pd
import numpy as np
from function.fn import *
import seaborn as sns
import io
import matplotlib.pyplot as plt



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


def handling_missing_value(data,cleaned_data):

    missing_value_technique=None
    missing_value_summary = None

    # data = st.session_state["data"]
    # cleaned_data = st.session_state["cleaned_data"] 

    # st.title("Handle Missing Values 🧩")

    # st.sidebar.header("Handle Missing Values")

    with st.container():
        st.header("Handel Missing values 🧩")

        sidebar = st.selectbox("sadas",["Missing Value Summary",'Missing Value Technique'])

        if sidebar == "Missing Value Summary":
            missing_value_summary = st.selectbox("Missing Value Summary",["Oreignal Data Summary","Cleaned Data Summary"])

        elif sidebar == "Missing Value Technique":
            missing_value_technique = st.selectbox("Handel Missing Values",
                                                        ['REMOVAL OF MISSING VALUES','IMPUTATION','FORWARD/BACKWARD FILL','PREDICTION MODELS'])

    # Original Data Summary
    if missing_value_summary == "Oreignal Data Summary":
        perform_eda(data)
        handle_missing_values(data)

    # Cleaned Data Summary
    elif missing_value_summary == "Cleaned Data Summary":
        perform_eda(cleaned_data)
        handle_missing_values(cleaned_data)



    #------------------ REMOVAL OF MISSING VALUES

    if missing_value_technique == "REMOVAL OF MISSING VALUES":
        cleaned_data = drop_columns_missing(cleaned_data)

    #------------------IMPUTATION 

    if missing_value_technique == "IMPUTATION":
        d_type = st.selectbox("Select Data Types", ["Number", "Object"])
        
        cleaned_data = impute_missing_values(cleaned_data, d_type)

    #------------------FORWARD/BACKWARD FILL

    if missing_value_technique == "FORWARD/BACKWARD FILL":

        # Get the list of columns with missing values in the DataFrame
        missing_columns = cleaned_data.columns

        # Allow user to select a column from the dropdown (selectbox) with missing values
        selected_column = st.multiselect("Select a column with missing values", missing_columns)

        # Allow user to select the method to fill missing values (forward fill or backward fill)
        fill_method = st.radio("Select method to fill missing values", ('ffill', 'bfill'))

        if selected_column:
            if st.button(f"Perform {fill_method}"):
                df_filled = fillMissingValues(cleaned_data, selected_column, fill_method)
                st.write("DataFrame after filling missing values:")
                st.write(df_filled)
        else:
            st.write("Please select a column with missing values.")

    #--------------------------PREDICTION MODELS
    if missing_value_technique=="PREDICTION MODELS":
        model_name = st.selectbox("Choose Model Name", ["Linear Regression","Random Forest"])
        st.info("work in progress")


    st.session_state["cleaned_data"] = cleaned_data
