from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
import streamlit as st
from sklearn.impute import SimpleImputer
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import hashlib

def oneHotEncoding(cleaned_data,selected_column):
    encoding = pd.get_dummies(cleaned_data[selected_column])
    cleaned_data.drop(columns=cleaned_data[selected_column], inplace=True)
    cleaned_data = pd.concat([cleaned_data, encoding], axis=1)
    return cleaned_data


def labelEncoding(cleaned_data,selected_columns):
    encoder = LabelEncoder()
    for column in selected_columns:
        cleaned_data[column] = encoder.fit_transform(cleaned_data[column])
    return cleaned_data


def ordinalEncoding(cleaned_data, selected_column, categories):
    cleaned_data[selected_column] = cleaned_data[selected_column].map(categories)
    return cleaned_data

def countingEncoding(cleaned_data, selected_column):
    count_encoding = cleaned_data[selected_column].value_counts().to_dict()
    cleaned_data[selected_column] = cleaned_data[selected_column].map(count_encoding)
    return cleaned_data

def featureHashing(cleaned_data, selected_data, n_features):
    for i in range(n_features):
        cleaned_data[f"{selected_data}_hash_{i}"] = cleaned_data[selected_data].apply(lambda x: int(hashlib.sha256(f"{x}_{i}".encode('utf-8')).hexdigest(), 16) % 10**8)
    cleaned_data.drop(columns=[selected_data], inplace=True)
    return cleaned_data

