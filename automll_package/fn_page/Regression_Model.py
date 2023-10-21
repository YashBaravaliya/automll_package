import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle
import os
import seaborn as sns

# Load your data here
# data = load_data()

data = st.session_state["data"]
cleaned_data = st.session_state["cleaned_data"]

def linear_visualization(X_test, y_test, y_pred):
    plt.scatter(X_test, y_test, color='b', label='Actual')
    plt.plot(X_test, y_pred, color='k', label='Predicted')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression')
    plt.legend()
    st.pyplot(plt)

def polynomial_visualization(X_test, y_test, y_pred, degree):
    sns.lmplot(X_test,y_pred,order = 2)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'Polynomial Regression (Degree {degree})')
    plt.legend()
    st.pyplot(plt)

def multiple_regression_visualization(X_test, y_test, y_pred):
    # Create separate scatter plots for each independent variable
    for column in X_test.columns:
        plt.scatter(X_test[column], y_test, label=f'Actual ({column})')
        plt.scatter(X_test[column], y_pred, label=f'Predicted ({column})', marker='x')
    
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Multiple Regression')
    plt.legend()
    st.pyplot(plt)

def lasso_visualization(X_test, y_test, y_pred, alpha):
    plt.scatter(X_test, y_test, color='b', label='Actual')
    plt.plot(X_test, y_pred, color='k', label='Predicted')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'Lasso Regression (Alpha {alpha})')
    plt.legend()
    st.pyplot(plt)

def ridge_visualization(X_test, y_test, y_pred, alpha):
    plt.scatter(X_test, y_test, color='b', label='Actual')
    plt.plot(X_test, y_pred, color='k', label='Predicted')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'Ridge Regression (Alpha {alpha})')
    plt.legend()
    st.pyplot(plt)

def regression():
    st.title("Regression Model Selection and Evaluation")

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
    regression_model = st.selectbox("Select Regression Model", ["Linear Regression", "Polynomial Regression", "Ridge", "Lasso", "Random Forest", "Decision Tree"])
    custom_params = None

    if regression_model == "Decision Tree":
        st.subheader("Custom Hyperparameters for Decision Tree")
        max_depth = st.number_input("Max Depth", min_value=1, value=5)
        min_samples_split = st.number_input("Min Samples Split", min_value=2, value=2)
        custom_params = {"max_depth": max_depth, "min_samples_split": min_samples_split}

    # Train the selected model with custom parameters
    if regression_model == "Linear Regression":
        model = LinearRegression()
    elif regression_model == "Polynomial Regression":
        degree = st.number_input("Degree of Polynomial Features", min_value=1, value=2)
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    elif regression_model == "Ridge":
        alpha = st.number_input("Alpha (Regularization Strength)", min_value=0.0, value=1.0)
        model = Ridge(alpha=alpha)
    elif regression_model == "Lasso":
        alpha = st.number_input("Alpha (Regularization Strength)", min_value=0.0, value=1.0)
        model = Lasso(alpha=alpha)
    elif regression_model == "Random Forest":
        n_estimators = st.number_input("Number of Estimators", min_value=1, value=100)
        model = RandomForestRegressor(n_estimators=n_estimators)
    elif regression_model == "Decision Tree":
        model = DecisionTreeRegressor(**custom_params)

    # Train the selected model
    if st.button("Train Model"):
        model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_train_pred = model.predict(X_train_scaled)

        st.write(regression_model)

        # Calculate and display evaluation metrics
        mse_test = mean_squared_error(y_test, y_pred)
        r2_test = r2_score(y_test, y_pred)
        accuracy_test_percentage = r2_test * 100

        mse_train = mean_squared_error(y_train, y_train_pred)
        r2_train = r2_score(y_train, y_train_pred)
        accuracy_train_percentage = r2_train * 100

        st.success(f"Test Accuracy: {accuracy_test_percentage:.2f}%")
        st.warning(f"Train Accuracy: {accuracy_train_percentage:.2f}%")

        if regression_model == "Linear Regression":
            linear_visualization(X_test,y_test,y_pred)
        if regression_model == "Polynomial Regression":
            linear_visualization(X_test,y_test,y_pred)
        if regression_model == "Ridge":
            linear_visualization(X_test,y_test,y_pred)
        if regression_model == "Lasso":
            linear_visualization(X_test,y_test,y_pred)
        if regression_model == "Random Forest":
            linear_visualization(X_test,y_test,y_pred)

        if regression_model == "Decision Tree":
            st.subheader("Decision Tree Visualization")
            st.write("The decision tree diagram is displayed below:")
            plot_tree(model, feature_names=X.columns, filled=True, rounded=True)
            st.pyplot(plt)

    # Download the trained model in pickle format to a custom path
    custom_path = st.text_input("Enter custom path to save the model (optional)")
    if st.button("Download Model"):
        model_filename = f"{regression_model}_model.pkl"
        if custom_path:
            custom_path = os.path.normpath(custom_path)
            model_filename = os.path.join(custom_path, model_filename)
        with open(model_filename, "wb") as model_file:
            pickle.dump(model, model_file)
        st.success("Model save Successfully")
            


