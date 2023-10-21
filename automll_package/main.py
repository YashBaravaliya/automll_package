import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import seaborn as sns
import warnings
from streamlit_option_menu import option_menu
from fn_page.upload import upload
from fn_page.Feature_Engineering import fe
from fn_page.Regression_Model import regression
from fn_page.Classification_Model import classification
from fn_page.Clustering_Model import clustering


# warnings.simplefilter("ignore", UserWarning)
st.set_option('deprecation.showPyplotGlobalUse', False)

def main():


selected = option_menu(
    menu_title = "AutoMll: Instant Machine Learning",
    options = ['Upload_Data','Feature_Enginerring','Regression','Classification','Clustering'],
    orientation="horizontal",
    icons=['cloud-upload-fill','clipboard-data-fill','bar-chart-line-fill','collection-fill','vr'],
    menu_icon='pie-chart-fill',

)

# Function to perform some exploratory data analysis


if selected == 'Upload_Data':
    upload()

elif selected == 'Feature_Enginerring':
    fe()

elif selected == 'Regression':
    regression()

elif selected == 'Classification':
    classification()

elif selected == 'Clustering':
    clustering()

if __name__ == "__main__": 
    main()
