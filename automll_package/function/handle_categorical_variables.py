import streamlit as st
import pandas as pd
import numpy as np
from function.encoding import *
import seaborn as sns
import matplotlib.pyplot as plt

# Function to display tooltips
def show_tooltip(text, tooltip_text):
    st.write(f'<span title="{tooltip_text}">{text}</span>', unsafe_allow_html=True)

cleaned_data = st.session_state["cleaned_data"]
data = st.session_state["data"]

def handling_categorical_values(data,cleaned_data):

    with st.container():
        st.header("Handle Categorical Variables 🧩")
        # st.title("Handle Categorical Variables 🧩")

        encoding_options = {
            "One-Hot Encoding": "Each category becomes a new binary column (dummy variable).",
            "Label Encoding": "Assigns a unique integer to each category. Suitable for ordinal data.",
            "Ordinal Encoding": "Maps categories to ordered integers based on user-defined order.",
            "Count Encoding": "Replaces categories with their respective frequency counts.",
            "Feature Hashing": "Hashes categories into a fixed number of features. Useful for high cardinality data.",
        }

        encoding = st.selectbox("Categorical Encoding", list(encoding_options.keys()))

        # Display tooltips for encoding options
        show_tooltip("ℹ️", encoding_options[encoding])

        object_columns = cleaned_data.select_dtypes(include=["object"])
        st.dataframe(object_columns)


    if encoding == "One-Hot Encoding":
        object_columns = cleaned_data.select_dtypes(include=["object"])
        # st.dataframe(object_columns)
        selected_columns = st.multiselect("Select Columns",object_columns.columns)

        # Display count of each unique value from the selected column
        if selected_columns:
            value_counts = cleaned_data[selected_columns].value_counts()
            st.write("Count of each unique value in the selected column:")
            st.write(value_counts)

        if st.button("One-Hot Encoding"):
            if selected_columns:
                cleaned_data =  oneHotEncoding(cleaned_data,selected_columns)
                st.write("DataFrame after One-Hot Encoding")
                st.write(cleaned_data)
            else:
                st.write("Please select column for One-Hot Encoding")

    elif encoding == "Label Encoding":
        object_columns = cleaned_data.select_dtypes(include=["object"])
        # st.dataframe(object_columns)
        selected_columns = st.multiselect("Select Columns",object_columns.columns)

        # Display count of each unique value from the selected column
        if selected_columns:
            value_counts = cleaned_data[selected_columns].value_counts()
            st.write("Count of each unique value in the selected column:")
            st.write(value_counts)

        if st.button("Label Encoding"):
            if selected_columns:
                cleaned_data =  labelEncoding(cleaned_data,selected_columns)
                st.write("DataFrame after Label encoding:")
                st.write(cleaned_data)
            else:
                st.write("Please select a column for Label Encoding")

    elif encoding == "Ordinal Encoding":

        # Create an empty dictionary for ordinal mapping
        ordinal_mapping = {}

        columns_list = cleaned_data.select_dtypes(include=['object']).columns.tolist()

        # Allow user to select a column from the dropdown (selectbox)
        selected_column = st.selectbox("Select a column", columns_list)

        # Display unique values from the selected column
        if selected_column:
            unique_values = cleaned_data[selected_column].unique()
            st.write("Unique values in the selected column:")
            st.write(unique_values)

            # Allow user to input numeric ordinal mapping for each unique value
            for value in unique_values:
                ordinal_mapping[value] = st.number_input(f"Enter ordinal mapping for '{value}'", min_value=0, step=1)


        # Perform ordinal encoding based on custom mapping
        if st.button("Perform Ordinal Encoding"):
            if selected_column:
                df_encoded = ordinalEncoding(cleaned_data, selected_column, ordinal_mapping)
                st.write("DataFrame after ordinal encoding:")
                st.write(df_encoded)
            else:
                st.write("Please select a column for ordinal encoding.")

    elif encoding == "Count Encoding":

        # Get the list of unique columns in the DataFrame
        columns_list = cleaned_data.select_dtypes(include=['object']).columns.tolist()

        # Allow user to select a column from the dropdown (selectbox)
        selected_column = st.selectbox("Select a column", columns_list)

        # Display count of each unique value from the selected column
        if selected_column:
            value_counts = cleaned_data[selected_column].value_counts()
            st.write("Count of each unique value in the selected column:")
            st.write(value_counts)

        # Perform counting encoding
        if st.button("Perform Counting Encoding"):
            if selected_column:
                df_encoded = countingEncoding(cleaned_data, selected_column)
                st.write("DataFrame after counting encoding:")
                st.write(df_encoded)
            else:
                st.write("Please select a column for counting encoding.")

    elif encoding == "Feature Hasing":
        object_columns = cleaned_data.select_dtypes(include=["object"])
        selected_columns = st.selectbox("Select Columns",object_columns.columns)

        # Allow user to input the number of hash features
        n_features = st.number_input("Enter the number of hash features", min_value=1, step=1)

        # Perform Feature Hashing
        if st.button("Perform Feature Hashing"):
            if selected_columns and n_features:
                df_encoded = featureHashing(cleaned_data, selected_columns, n_features)
                st.write("DataFrame after Feature Hashing:")
                st.write(df_encoded)
            else:
                st.write("Please select a column and enter the number of hash features.")


    st.session_state["cleaned_data"] = cleaned_data
    