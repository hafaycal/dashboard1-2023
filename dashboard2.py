# TO RUN : $streamlit run dashboard/dashboard.py
# Local URL: http://localhost:8501
# Network URL: http://192.168.0.50:8501
# Online URL : http://15.188.179.79

#

import streamlit as st

# Utilisation de SK_IDS dans st.sidebar.selectbox
import seaborn as sns
import os
import plotly.express as px
import streamlit as st
from PIL import Image
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import time
import pickle
import plotly.graph_objects as go
import shap
from streamlit_shap import st_shap
import numpy as np
###---------- load data -------- 



def get_sk_id_list():
        # API_URL = "http://127.0.0.1:5000/api/"
        API_URL = "https://api12023.herokuapp.com/"
        
        # URL of the sk_id API
        SK_IDS_API_URL = API_URL + "sk_ids/"

        # Requesting the API and saving the response
        response = requests.get(SK_IDS_API_URL)

        # Convert from JSON format to Python dict
        content = json.loads(response.content)

        # Getting the values of SK_IDS from the content
        SK_IDS = content['data']
        print ('MESSAGE 1')
        return SK_IDS
    
        



def main():

    SK_IDS = get_sk_id_list()

    # Logo "Prêt à dépenser"
    image = Image.open('logo.png')
    st.sidebar.image(image, width=280)
    st.title('Tableau de bord - "Prêt à dépenser"')

    ### Title
    st.title('Home Credit Default Risk')

    ##################################################
    # Selecting applicant ID
    select_sk_id = st.sidebar.selectbox('Select SK_ID from list:', SK_IDS, key=1)
    st.write('You selected: ', select_sk_id)


#################################################################

from app import app as application
if __name__ == "__main__":

    print ('MESSAGE 2')

    # Titre 1
    st.markdown("""
                <h1 style="color:#772b58;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
                1. Quel est le score de votre client ?</h1>
                """, 
                unsafe_allow_html=True)
    st.write("")




