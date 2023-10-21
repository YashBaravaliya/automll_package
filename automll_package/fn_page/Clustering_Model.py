import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, AffinityPropagation, SpectralClustering
import matplotlib.pyplot as plt
import pickle
import os

# Load your data here
# data = load_data()

data = st.session_state["data"]
cleaned_data = st.session_state["cleaned_data"]

def clustering():
    st.title("Clustering Model Selection and Visualization")

    st.subheader("Select Features for Clustering")
    selected_features = st.multiselect("Select the Features for Clustering", data.columns)

    # Create X based on the selected features
    X = cleaned_data[selected_features]

    st.subheader("Model Selection")
    clustering_model = st.selectbox("Select Clustering Algorithm", ["K-Means", "Agglomerative Clustering", "DBSCAN", "Affinity Propagation", "Spectral Clustering"])
    custom_params = None

    if clustering_model == "K-Means":
        n_clusters = st.number_input("Number of Clusters (K)", min_value=2, value=4)
        model = KMeans(n_clusters=n_clusters, random_state=0)
    elif clustering_model == "Agglomerative Clustering":
        n_clusters = st.number_input("Number of Clusters", min_value=2, value=4)
        linkage = st.selectbox("Linkage", ["ward", "complete", "average", "single"])
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    elif clustering_model == "DBSCAN":
        eps = st.number_input("Epsilon (eps)", min_value=0.1, value=0.5)
        min_samples = st.number_input("Min Samples", min_value=1, value=5)
        model = DBSCAN(eps=eps, min_samples=min_samples)
    elif clustering_model == "Affinity Propagation":
        damping = st.number_input("Damping (0.5 to 1.0)", min_value=0.5, max_value=1.0, value=0.5)
        model = AffinityPropagation(damping=damping)
    elif clustering_model == "Spectral Clustering":
        n_clusters = st.number_input("Number of Clusters", min_value=2, value=4)
        affinity = st.selectbox("Affinity", ["rbf", "nearest_neighbors"])
        model = SpectralClustering(n_clusters=n_clusters, affinity=affinity)

    # Cluster the data
    labels = model.fit_predict(X)

    # Data visualization based on the clustering model
    if clustering_model != "DBSCAN":
        st.subheader("Clustering Visualization")
        plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='rainbow')
        plt.xlabel(selected_features[0])
        plt.ylabel(selected_features[1])
        plt.title(f"{clustering_model} Clustering")
        st.pyplot(plt)

    st.subheader("Cluster Labels")
    st.write(labels)

    # Download the clustering model in pickle format to a custom path
    custom_path = st.text_input("Enter custom path to save the model (optional)")
    if st.button("Download Model"):
        model_filename = f"{clustering_model}_model.pkl"
        if custom_path:
            custom_path = os.path.normpath(custom_path)
            model_filename = os.path.join(custom_path, model_filename)
        with open(model_filename, "wb") as model_file:
            pickle.dump(model, model_file)
        st.success("Model save Successfully")
        


