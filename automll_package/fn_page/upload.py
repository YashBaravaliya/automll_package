import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import seaborn as sns
import warnings
from streamlit_option_menu import option_menu

def perform_eda(data):
    st.subheader("Exploratory Data Analysis 📊")

    # Show data summary
    st.write("Data Summary 📈:")
    st.dataframe(data.describe())

    # Show data info
    st.write("Data Info ℹ️:")
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    # Show data distribution
    st.write("Data Distribution 📈:")
    for col in data.columns:
        fig = px.histogram(data, x=col, title=f"Distribution of {col}", labels={col: f"{col} Value"})
        
        # Customize the border style
        fig.update_traces(marker=dict(line=dict(width=1, color='Black')))
        
        st.plotly_chart(fig)

def upload():
    # st.set_page_config(page_title="Exploratory Data Analysis App", layout="wide")

    st.title("Upload Data 🚀")

    # upload_file = "titanic.csv"

    # st.sidebar.header("Feature Engineering")
    upload_file_ = st.file_uploader("Upload a CSV file 📂", type=["csv"])

    if upload_file_ is not None:
        data = pd.read_csv(upload_file_)
        cleaned_data = data.copy()

        # Save the cleaned data in session state
        st.session_state["cleaned_data"] = cleaned_data
        st.session_state["data"] = data
        st.session_state["file_name"] = upload_file_.name

        # Display the selected DataFrame in the main area
        st.subheader("Original DataFrame")

        # Show the uploaded data
        with st.container():
            st.subheader("Uploaded Data")
            st.write(data, wide_mode=True)

        with st.container():
            st.write("---")
            left_column, right_column = st.columns(2)

            # Perform Exploratory Data Analysis
            with left_column:
                perform_eda(data)

            # Perform Feature Engineering (add more feature engineering options if needed)
            with right_column:
                # st.subheader("Data Visualization Options")

                st.subheader("Seaborn Data Visualization")

                # Dropdown for selecting the type of plot
                plot_type = st.selectbox("Select a Plot Type", ["Scatter Plot", "Line Plot", "Histogram", "Box Plot", "Violin Plot", "Pair Plot", "Heatmap"])

                if plot_type == "Scatter Plot":
                    x_column = st.selectbox("Select X-Axis Column", data.columns)
                    y_column = st.selectbox("Select Y-Axis Column", data.columns)
                    sns.scatterplot(data=data, x=x_column, y=y_column)
                    st.pyplot()
                elif plot_type == "Line Plot":
                    x_column = st.selectbox("Select X-Axis Column", data.columns)
                    y_column = st.selectbox("Select Y-Axis Column", data.columns)
                    sns.lineplot(data=data, x=x_column, y=y_column)
                    st.pyplot()
                elif plot_type == "Histogram":
                    column = st.selectbox("Select a Column", data.columns)
                    sns.histplot(data=data, x=column, kde=True)
                    st.pyplot()
                elif plot_type == "Box Plot":
                    x_column = st.selectbox("Select X-Axis Column", data.columns)
                    y_column = st.selectbox("Select Y-Axis Column", data.columns)
                    sns.boxplot(data=data, x=x_column, y=y_column)
                    st.pyplot()
                elif plot_type == "Violin Plot":
                    x_column = st.selectbox("Select X-Axis Column", data.columns)
                    y_column = st.selectbox("Select Y-Axis Column", data.columns)
                    sns.violinplot(data=data, x=x_column, y=y_column)
                    st.pyplot()
                elif plot_type == "Pair Plot":
                    sns.pairplot(data)
                    st.pyplot()
                elif plot_type == "Heatmap":
                    sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
                    st.pyplot()

        # Save the processed dataset to disk for later use
