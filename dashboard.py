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

#------------- Affichage des infos client en HTML------------------------------------------
def display_client_info(id,revenu,age,nb_ann_travail):
   
    components.html(
    """
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    <div class="card" style="width: 500px; margin:10px;padding:0">
        <div class="card-body">
            <h5 class="card-title">Info Client</h5>
            
            <ul class="list-group list-group-flush">
                <li class="list-group-item"> <b>ID                           : </b>"""+id+"""</li>
                <li class="list-group-item"> <b>Revenu                       : </b>"""+revenu+"""</li>
                <li class="list-group-item"> <b>Age                          : </b>"""+age+"""</li>
                <li class="list-group-item"> <b>Nombre d'années travaillées  : </b>"""+nb_ann_travail+"""</li>
            </ul>
        </div>
    </div>
    """,
    height=300
    
    )
#==============================================================================================
def predict():
    lgbm = pickle.load(open("lgbm.pkl", 'rb'))
    if lgbm:
        try:
            json_ = request.json
            print(json_)
            query = pd.DataFrame(json_)           
            #X_transformed = preprocessing(query)
            y_pred = randomForest.predict(X_train)
            y_proba = randomForest.predict_proba(X_train)
            
            return jsonify({'prediction': y_pred,'prediction_proba':y_proba[0][0]})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Problem loading the model')
        return ('No model here to use')

def predictByClientId():
    lgbm = pickle.load(open("lgbm.pkl", 'rb'))
    if lgbm:
        try:
            json_ = request.json
            print(json_)
            sample_size = 10000
            
            print(json_)  

            sample_size= 20000
            #data_set = data = pd.read_csv("df_final.csv",nrows=sample_size)
            client=data_set[X_train['SK_ID_CURR']==json_['SK_ID_CURR']].drop(['SK_ID_CURR','TARGET'],axis=1)
            print(client)
 
            
            preproc = pickle.load(open("preprocessor.sav", 'rb'))
            #X_transformed =preproc.transform(client)
            y_pred = randomForest.predict(client)
            y_proba = randomForest.predict_proba(client)
            
            return jsonify({'prediction': str(y_pred[0]),'prediction_proba':str(y_proba[0][0])})


        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Problem loading the model')
        return ('No model here to use')


def load_dataset(sample_size):
    df_final = pd.read_csv("data/df_final.csv",nrows=sample_size)
    return df_final

'''Load fitted preprocessor '''
def load_preprocessor():
    preproc = pickle.load(open("models/preprocessor.sav", 'rb'))
    return preproc


'''Charger un modele entrainné '''
def load_model(model_to_load):
    if model_to_load == "randomForest":
        model = pickle.load(open("models/classifier_rf_model.sav", 'rb'))
    elif model_to_load == "lgbm":
        model = pickle.load(open("models/model_LGBM.pkl", 'rb'))
    else:
        print("modèle non connu ! Merci de chois : lgbm ou randomForest")
    
    return model

'''Prédire un client avec un modele  '''
def predict_client(model,X):
    X = X.drop(['SK_ID_CURR'],axis=1)
    #X_transformed = preprocessing(X)
    model = load_model(model)
    
    #y_pred = model.predict(X_transformed)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    #y_proba = model.predict_proba(X)

    return y_pred,y_proba


'''Prédire un client par son ID dans le dataset '''
def predict_client_par_ID(model_to_use,id_client):
    sample_size= 2000
    data_set = load_dataset(sample_size)
    client=data_set[data_set['SK_ID_CURR']==id_client].drop(['SK_ID_CURR','TARGET'],axis=1)
    print(client)
    #client_preproceced = preprocessing(client)
    model = load_model(model_to_use)

    y_pred = model.predict(client)
    #y_pred = model.predict(client)
    
    y_proba = model.predict_proba(client)
    #y_proba = model.predict_proba(client)
    return y_pred,y_proba

def get_sk_id_list():
        # API_URL = "http://127.0.0.1:5000/api/"
        API_URL = "https://api5-p7.herokuapp.com/api/"
        
        # URL of the sk_id API
        SK_IDS_API_URL = API_URL + "sk_ids/"

        # Requesting the API and saving the response
        response = requests.get(SK_IDS_API_URL)

        # Convert from JSON format to Python dict
        content = json.loads(response.content)

        # Getting the values of SK_IDS from the content
        SK_IDS = content['data']

        return SK_IDS
    
        

    ##################################################
    ##################################################
    ##################################################

### Data
def show_data(data):
    st.write(data.head(10))

    print("je suis dans la fonction")
### Solvency
def pie_chart(thres,data):
    #st.write(100* (data['TARGET']>thres).sum()/data.shape[0])
    print("je suis dans la fonction")
    percent_sup_seuil =100* (data['TARGET']>thres).sum()/data.shape[0]
    percent_inf_seuil = 100-percent_sup_seuil
    d = {'col1': [percent_sup_seuil,percent_inf_seuil], 'col2': ['% Non Solvable','% Solvable',]}
    df = pd.DataFrame(data=d)
    #fig = plt.pie(df,values='col1', names='col2', title=' Pourcentage de solvabilité des clients di dataset')
    fig = plt.pie(df)
    #plt.pie(y, labels = mylabels, colors=colors,explode = explodevalues, autopct='%1.1f%%', shadow = True)

    st.plotly_chart(fig)

def show_overview(data):
    st.title("Risque")
    risque_threshold = st.slider(label = 'Seuil de risque', min_value = 0.0,
                    max_value = 1.0 ,
                     value = 0.5,
                     step = 0.1)
    #st.write(risque_threshold)
    pie_chart(risque_threshold,data) 

 
### Graphs
def filter_graphs():
    st.subheader("Filtre des Graphes")
    col1, col2,col3 = st.columns(3)
    is_educ_selected = col1.radio("Graph Education",('non','oui'))
    is_statut_selected = col2.radio('Graph Statut',('non','oui'))
    is_income_selected = col3.radio('Graph Revenu',('non','oui'))

    return is_educ_selected,is_statut_selected,is_income_selected

def hist_graph ():
    st.bar_chart(data['DAYS_BIRTH'])
    df = pd.DataFrame(data[:200],columns = ['DAYS_BIRTH','AMT_CREDIT'])
    df.hist()
    st.pyplot()

def education_type(train_set):
    ed = train_set.groupby('NAME_EDUCATION_TYPE').NAME_EDUCATION_TYPE.count()
    u_ed = train_set.NAME_EDUCATION_TYPE.unique() 
    fig = plt.bar(u_ed, ed, bottom=None, color='blue', label='Education')
    st.plotly_chart(fig)


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

# Ajout de données à data

    if not data.empty:
        st.write(data.head(10))
    else:
        st.write("Le DataFrame est vide.")    

def lecture_X_test_original():
    X_test_original = pd.read_csv("Data/X_test_original.csv")
    X_test_original = X_test_original.rename(columns=str.lower)
    return X_test_original

def lecture_X_test_clean():
    X_test_clean = pd.read_csv("Data/X_test_clean.csv")
    #st.dataframe(X_test_clean)
    return X_test_clean

def lecture_description_variables():
    description_variables = pd.read_csv("Data/description_variable.csv", sep=";")
    return description_variables


if __name__ == "__main__":

    lecture_X_test_original()
    lecture_X_test_clean()
    lecture_description_variables()

    # Titre 1
    st.markdown("""
                <h1 style="color:#772b58;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
                1. Quel est le score de votre client ?</h1>
                """, 
                unsafe_allow_html=True)
    st.write("")

    ##########################################################
    # Création et affichage du sélecteur du numéro de client #
    ##########################################################
    liste_clients = list(lecture_X_test_original()['sk_id_curr'])
    col1, col2 = st.columns(2) # division de la largeur de la page en 2 pour diminuer la taille du menu déroulant
    with col1:
        ID_client = st.selectbox("*Veuillez sélectionner le numéro de votre client à l'aide du menu déroulant 👇*", 
                                (liste_clients))
        st.write("Vous avez sélectionné l'identifiant n° :", ID_client)
    with col2:
        st.write("")

    #################################################
    # Lecture du modèle de prédiction et des scores #
    #################################################
    model_LGBM = pickle.load(open("model_LGBM.pkl", "rb"))
    y_pred_lgbm = model_LGBM.predict(lecture_X_test_clean().drop(labels="sk_id_curr", axis=1))    # Prédiction de la classe 0 ou 1
    y_pred_lgbm_proba = model_LGBM.predict_proba(lecture_X_test_clean().drop(labels="sk_id_curr", axis=1)) # Prédiction du % de risque

    # Récupération du score du client
    y_pred_lgbm_proba_df = pd.DataFrame(y_pred_lgbm_proba, columns=['proba_classe_0', 'proba_classe_1'])
    y_pred_lgbm_proba_df = pd.concat([y_pred_lgbm_proba_df['proba_classe_1'],
                                    lecture_X_test_clean()['sk_id_curr']], axis=1)
    #st.dataframe(y_pred_lgbm_proba_df)
    score = y_pred_lgbm_proba_df[y_pred_lgbm_proba_df['sk_id_curr']==ID_client]
    score_value = round(score.proba_classe_1.iloc[0]*100, 2)

    # Récupération de la décision
    y_pred_lgbm_df = pd.DataFrame(y_pred_lgbm, columns=['prediction'])
    y_pred_lgbm_df = pd.concat([y_pred_lgbm_df, lecture_X_test_clean()['sk_id_curr']], axis=1)
    y_pred_lgbm_df['client'] = np.where(y_pred_lgbm_df.prediction == 1, "non solvable", "solvable")
    y_pred_lgbm_df['decision'] = np.where(y_pred_lgbm_df.prediction == 1, "refuser", "accorder")
    solvabilite = y_pred_lgbm_df.loc[y_pred_lgbm_df['sk_id_curr']==ID_client, "client"].values
    decision = y_pred_lgbm_df.loc[y_pred_lgbm_df['sk_id_curr']==ID_client, "decision"].values

    ##############################################################
    # Affichage du score et du graphique de gauge sur 2 colonnes #
    ##############################################################
    col1, col2 = st.columns(2)
    with col2:
        st.markdown(""" <br> <br> """, unsafe_allow_html=True)
        st.write(f"Le client dont l'identifiant est **{ID_client}** a obtenu le score de **{score_value:.1f}%**.")
        st.write(f"**Il y a donc un risque de {score_value:.1f}% que le client ait des difficultés de paiement.**")
        st.write(f"Le client est donc considéré par *'Prêt à dépenser'* comme **{solvabilite[0]}** \
                et décide de lui **{decision[0]}** le crédit. ")
    # Impression du graphique jauge
    with col1:
        fig = go.Figure(go.Indicator(
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        value = float(score_value),
                        mode = "gauge+number+delta",
                        title = {'text': "Score du client", 'font': {'size': 24}},
                        delta = {'reference': 35.2, 'increasing': {'color': "#3b203e"}},
                        gauge = {'axis': {'range': [None, 100],
                                'tickwidth': 3,
                                'tickcolor': 'darkblue'},
                                'bar': {'color': 'white', 'thickness' : 0.3},
                                'bgcolor': 'white',
                                'borderwidth': 1,
                                'bordercolor': 'gray',
                                'steps': [{'range': [0, 20], 'color': '#e8af92'},
                                        {'range': [20, 40], 'color': '#db6e59'},
                                        {'range': [40, 60], 'color': '#b43058'},
                                        {'range': [60, 80], 'color': '#772b58'},
                                        {'range': [80, 100], 'color': '#3b203e'}],
                                'threshold': {'line': {'color': 'white', 'width': 8},
                                            'thickness': 0.8,
                                            'value': 35.2 }}))

        fig.update_layout(paper_bgcolor='white',
                        height=400, width=500,
                        font={'color': '#772b58', 'family': 'Roboto Condensed'},
                        margin=dict(l=30, r=30, b=5, t=5))
        st.plotly_chart(fig, use_container_width=True)

    ################################
    # Explication de la prédiction #
    ################################
    # Titre 2
    st.markdown("""
                <h1 style="color:#772b58;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
                2. Comment le score de votre client est-il calculé ?</h1>
                """, 
                unsafe_allow_html=True)
    st.write("")

    # Calcul des valeurs Shap
    explainer_shap = shap.TreeExplainer(model_LGBM)
    shap_values = explainer_shap.shap_values(lecture_X_test_clean().drop(labels="sk_id_curr", axis=1))

    # récupération de l'index correspondant à l'identifiant du client
    idx = int(lecture_X_test_clean()[lecture_X_test_clean()['sk_id_curr']==ID_client].index[0])

    # Graphique force_plot
    st.write("Le graphique suivant appelé `force-plot` permet de voir où se place la prédiction (f(x)) par rapport à la `base value`.") 
    st.write("Nous observons également quelles sont les variables qui augmentent la probabilité du client d'être \
            en défaut de paiement (en rouge) et celles qui la diminuent (en bleu), ainsi que l’amplitude de cet impact.")
    st_shap(shap.force_plot(explainer_shap.expected_value[1], 
                            shap_values[1][idx,:], 
                            lecture_X_test_clean().drop(labels="sk_id_curr", axis=1).iloc[idx,:], 
                            link='logit',
                            figsize=(20, 8),
                            ordering_keys=True,
                            text_rotation=0,
                            contribution_threshold=0.05))
    # Graphique decision_plot
    st.write("Le graphique ci-dessous appelé `decision_plot` est une autre manière de comprendre la prédiction.\
            Comme pour le graphique précédent, il met en évidence l’amplitude et la nature de l’impact de chaque variable \
            avec sa quantification ainsi que leur ordre d’importance. Mais surtout il permet d'observer \
            “la trajectoire” prise par la prédiction du client pour chacune des valeurs des variables affichées. ")
    st.write("Seules les 15 variables explicatives les plus importantes sont affichées par ordre décroissant.")
    st_shap(shap.decision_plot(explainer_shap.expected_value[1], 
                            shap_values[1][idx,:], 
                            lecture_X_test_clean().drop(labels="sk_id_curr", axis=1).iloc[idx,:], 
                            feature_names=lecture_X_test_clean().drop(labels="sk_id_curr", axis=1).columns.to_list(),
                            feature_order='importance',
                            feature_display_range=slice(None, -16, -1), # affichage des 15 variables les + importantes
                            link='logit'))

    # Titre 3
    st.markdown("""
                <h1 style="color:#772b58;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
                3. Lexique des variables </h1>
                """, 
                unsafe_allow_html=True)
    st.write("")

    st.write("La base de données globale contient un peu plus de 200 variables explicatives. Certaines d'entre elles étaient peu \
            renseignées ou peu voir non disciminantes et d'autres très corrélées (2 variables corrélées entre elles \
            apportent la même information : l'une d'elles est donc redondante).")
    st.write("Après leur analyse, 56 variables se sont avérées pertinentes pour prédire si le client aura ou non des difficultés de paiement.")

    pd.set_option('display.max_colwidth', None)
    st.dataframe(lecture_description_variables())



def test_load():
    assert load_dataset(1).size == 500  
