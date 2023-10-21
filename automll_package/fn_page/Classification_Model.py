import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import pickle
import os

# Load your data here
# data = load_data()

data = st.session_state["data"]
cleaned_data = st.session_state["cleaned_data"]

def classification():
    st.title("Classification Model Selection and Evaluation")

    st.subheader("Select Target Variable")
    target_variable = st.selectbox("Select the Target Variable", data.columns)

    # Create X and y based on the selected target variable
    y = cleaned_data[target_variable]
    X = cleaned_data.drop(target_variable, axis=1)

    st.subheader("1. Train-Test Split Configuration")
    test_size = st.slider("Test Size (Percentage)", 0.01, 0.5, 0.2)
    random_state = st.number_input("Random State", value=42)

    # Apply train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    st.subheader("2. Data Scaling")
    scaling_technique = st.selectbox("Select Scaling Technique", ["Standardization", "Min-Max Scaling"])
    if scaling_technique == "Standardization":
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    elif scaling_technique == "Min-Max Scaling":
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

    st.subheader("3. Model Selection")
    classification_model = st.selectbox("Select Classification Model", ["Random Forest", "K-Nearest Neighbors (KNN)", "Naive Bayes", "SVM", "Logistic Regression", "Decision Tree"])
    custom_params = None

    if classification_model == "Decision Tree":
        st.subheader("Custom Hyperparameters for Decision Tree")
        max_depth = st.number_input("Max Depth", min_value=1, value=5)
        min_samples_split = st.number_input("Min Samples Split", min_value=2, value=2)
        custom_params = {"max_depth": max_depth, "min_samples_split": min_samples_split}

    if classification_model == "SVM":
        C = st.number_input("Regularization Parameter (C)", min_value=0.001, value=1.0)
        kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
        model = SVC(C=C, kernel=kernel)

    if classification_model == "Logistic Regression":
        C = st.number_input("Regularization Parameter (C)", min_value=0.001, value=1.0)
        model = LogisticRegression(C=C)

    if classification_model == "Random Forest":
        n_estimators = st.number_input("Number of Estimators", min_value=1, value=100)
        model = RandomForestClassifier(n_estimators=n_estimators)

    if classification_model == "K-Nearest Neighbors (KNN)":
        n_neighbors = st.number_input("Number of Neighbors", min_value=1, value=5)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)

    if classification_model == "Naive Bayes":
        model = GaussianNB()

    # Train the selected model with custom parameters
    if classification_model == "Decision Tree":
        model = DecisionTreeClassifier(**custom_params)

    # Train the selected model
    if st.button("Train Model"):
        model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_train_pred = model.predict(X_train_scaled)

        # Calculate and display evaluation metrics
        accuracy_test = accuracy_score(y_test, y_pred)
        classification_report_test = classification_report(y_test, y_pred)

        accuracy_train = accuracy_score(y_train, y_train_pred)
        classification_report_train = classification_report(y_train, y_train_pred)

        st.success(f"Test Accuracy: {accuracy_test:.2f}")
        st.warning(f"Train Accuracy: {accuracy_train:.2f}")

        st.subheader("Classification Report for Test Data:")
        st.text(classification_report_test)

        st.subheader("Classification Report for Train Data:")
        st.text(classification_report_train)

    # Download the trained model in pickle format to a custom path
    custom_path = st.text_input("Enter custom path to save the model (optional)")
    if st.button("Download Model"):
        model_filename = f"{classification_model}_model.pkl"
        if custom_path:
            custom_path = os.path.normpath(custom_path)
            model_filename = os.path.join(custom_path, model_filename)
        with open(model_filename, "wb") as model_file:
            pickle.dump(model, model_file)
        st.success("Model save Successfully")
            


