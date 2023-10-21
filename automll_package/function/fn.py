import streamlit as st
from sklearn.impute import SimpleImputer
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Function to handle missing values
def handle_missing_values(cleaned_data):
    # Plot a missing value heatmap
    st.subheader("Missing Values Heatmap")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cleaned_data.isnull(), cmap='coolwarm')
    plt.title("Missing Values Heatmap")
    st.pyplot(fig)

    # Display missing value count per column
    st.subheader("Missing Value Count per Column")
    missing_counts = pd.DataFrame(cleaned_data.isnull().sum()).T
    st.write(missing_counts)

    # Bar plot of missing value percentage per column
    st.subheader("Missing Value Percentage per Column")
    missing_percentage = (cleaned_data.isnull().sum() / len(cleaned_data)) * 100
    fig, ax = plt.subplots(figsize=(8, 4))
    missing_percentage.plot(kind='bar', ax=ax)
    plt.ylabel("Percentage")
    plt.xlabel("Columns")
    plt.title("Missing Value Percentage per Column")
    st.pyplot(fig)

    # Histogram of missing value counts per row
    st.subheader("Missing Value Counts per Row")
    missing_counts_per_row = cleaned_data.isnull().sum(axis=1)
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.hist(missing_counts_per_row, bins=len(cleaned_data.columns))
    plt.xlabel("Missing Value Count")
    plt.ylabel("Frequency")
    plt.title("Missing Value Counts per Row")
    st.pyplot(fig)

    # Statistical summary of missing values
    st.subheader("Statistical Summary of Missing Values")
    missing_stats = cleaned_data.isnull().describe().T
    st.write(missing_stats)


def drop_columns_missing(df):
    percent=st.number_input("enter a percentage value(if your columns have more than percentage value it will drop the columns)",min_value=0.0, max_value=1.0, step=.01, format="%g")
    # Add a button to trigger the column dropping process
    if st.button("Drop Columns"):
        df_dropped = df.columns[df.isnull().mean() > percent]
        df1 = df.drop(df_dropped, axis=1)
        st.write(df1)
        st.text(f"{df_dropped[0]} columns dropped")
        return df1
    return df

def impute_missing_values(cleaned_data, d_type):
    if d_type == "Number":
        numeric_columns = cleaned_data.select_dtypes(include="number")
        # missing_columns = numeric_columns.columns[numeric_columns.isnull().any()].tolist()
        selected_columns = st.multiselect("Select columns", numeric_columns.columns)

        mean_median_mode = st.selectbox("Select Mean Median Mode", ["None", "Mean", "Median", "Mode","Fill with constant"])

        # Fill selected columns with their respective means
        if mean_median_mode == "Mean":
            for column in selected_columns:
                column_mean = cleaned_data[column].mean()
                cleaned_data[column].fillna(round(column_mean, 2), inplace=True)
        elif mean_median_mode == "Median":
            for column in selected_columns:
                column_median = cleaned_data[column].median()
                cleaned_data[column].fillna(column_median, inplace=True)
        elif mean_median_mode == "Mode":
            for column in selected_columns:
                column_mode = cleaned_data[column].mode().iloc[0]
                cleaned_data[column].fillna(column_mode, inplace=True)        

        elif mean_median_mode == "Fill with constant":
            constant = st.number_input("Enter a number constant")
            if st.button(f"Fill With {constant}"):
                for colummns in selected_columns:
                    cleaned_data[colummns].fillna(constant,inplace=True)
        st.write(cleaned_data)

    else:
        object_columns = cleaned_data.select_dtypes(include=["object"])
        missing_columns = object_columns.columns[object_columns.isnull().any()].tolist()
        selected_columns = st.multiselect("Select columns", missing_columns)

        if st.button("Mode"):
            for column in selected_columns:
                column_mode = cleaned_data[column].mode().iloc[0]
                cleaned_data[column].fillna(column_mode, inplace=True)
        st.write(cleaned_data)

    return cleaned_data

def fillMissingValues(cleaned_data, selected_columns, method):
    if method == 'ffill':
        cleaned_data[selected_columns].fillna(method='ffill', inplace=True)
    elif method == 'bfill':
        cleaned_data[selected_columns].fillna(method='bfill', inplace=True)
    return cleaned_data

# ------------------------------
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Function for Standardization (Z-score normalization)
def standardize_data(data):
    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit and transform the data
    scaled_data = scaler.fit_transform(data)

    # Convert the scaled data back to a DataFrame (assuming data is a DataFrame)
    scaled_df = pd.DataFrame(scaled_data, columns=data.columns)

    return scaled_df

# Function for Min-Max Scaling (Normalization)
def min_max_scale_data(data):
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Fit and transform the data
    scaled_data = scaler.fit_transform(data)

    # Convert the scaled data back to a DataFrame (assuming data is a DataFrame)
    scaled_df = pd.DataFrame(scaled_data, columns=data.columns)

    return scaled_df


def handling_scaling(data, cleaned_data):
    st.subheader("Data Scaling 📏")

    scaling_technique = st.selectbox("Select Scaling Technique", ["None","Standardization", "Min-Max Scaling"])

    if scaling_technique == "Standardization":
        cleaned_data = standardize_data(cleaned_data)
        st.write("Data after Standardization:")
        st.dataframe(cleaned_data)

    elif scaling_technique == "Min-Max Scaling":
        cleaned_data = min_max_scale_data(cleaned_data)
        st.write("Data after Min-Max Scaling:")
        st.dataframe(cleaned_data)

    elif scaling_technique =="None":
        pass
