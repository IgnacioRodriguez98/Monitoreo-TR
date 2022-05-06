import plotly.express as px
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import time
import numpy as np
from sklearn.ensemble import IsolationForest
from pyod.models.auto_encoder import AutoEncoder
from sklearn.preprocessing import StandardScaler


###################################Variables de lenguaje###########################################



with st.sidebar:
    option = st.selectbox(
     "Lenguage / Idioma" ,
     ('English', 'Español'))
##########################################################

st.write(""" # Introducción

Esta es una aplicación enfocada a la detección de anomalías en los gases que se generan dentro de un transformador; 
podemos realizar un análisis individual por cada gas, así como un análisis multivariado con 2 o más gases a la vez.

Los gases a analizar son los conocidos como "Gases Combustibles":
Acetileno, Hidrógeno, Etileno, Monóxido de Carbono, Etano y Metano.
""")

st.write(""" # ¿Qué es la detección de anomalías?
Como su nombre lo indica es localizar un comportamiento, patrón, información 
que este fuera de los límites “normales” 

## Detección de anomalias en tiempo real
La detección de anomalias en tiempo real en una planta ayudaría a evitar consecuencias graves, como lo podrían ser
paros imprevistos de los equipo hasta explosiones del equipo en cuestión. A su vez ayuda en la toma de decisiones más acertadas sobre qué hacer para no llegar a alguna consecuencia irreparable.

Esta aplicación simula ser un detector de anomalias en tiempo real instalado en una planta, antes de comenzar por favor seguir los siguientes pasos.""")

##########################################################

with st.sidebar:
    st.write("# Variables")
    database = st.radio(
     "Base de datos",
     ('Planta 1 (2,432 datos)', 'Planta 2 (9,277 datos)', 'Planta 3 (10,000 datos)'))

    st.write("# Gases:")
    C2H2 = st.checkbox("Acetileno")
    H2 = st.checkbox("Hidrógeno")
    C2H4 = st.checkbox("Etileno")
    CO = st.checkbox("Monóxido de carbono")
    C2H6 = st.checkbox("Etano")
    CH4 = st.checkbox("Metano")

if database == 'Planta 1 (2,432 datos)':
    #st.write("""# Has seleccionado la base de datos 1""")
    #df = pd.read_csv('https://docs.google.com/spreadsheets/d/1iuC6jPlFf5sbzM-E6u-L2FMQR8AjEpCQ0ZJyCpwTSJ4/edit?usp=sharing', header=None)
    df = pd.read_csv('https://raw.githubusercontent.com/IgnacioRodriguez98/Monitoreo-TR/main/Data/planta2.csv', header=None)
    #st.write(df)

elif database == 'Planta 2 (9,277 datos)':
    #st.write("""# Has seleccionado la base de datos 2""")
    #df= pd.read_csv('https://docs.google.com/spreadsheets/d/1-hiYI8sBIcoMXOvw0pCFOnnhrD-9f7ZW3A_EHuONlso/edit?usp=sharing', header=None)
    df = pd.read_csv('https://raw.githubusercontent.com/IgnacioRodriguez98/Monitoreo-TR/main/Data/planta3.csv', header=None)
    #st.write(df)


else:
    #st.write("""# Has seleccionado la base de datos 3""")
    #df= pd.read_csv('https://docs.google.com/spreadsheets/d/1R8YZq8Ex6YdRenFnisb7Ic_g1yRRoNQmmjUPxSbnufE/edit?usp=sharing', header=None)
    df = pd.read_csv('https://raw.githubusercontent.com/IgnacioRodriguez98/Monitoreo-TR/main/Data/planta1.csv', header=None)
    #st.write(df)
##################### SELECCION DE GASES ######################
st.write("""# 1er Paso: Selección de base de datos

En la parte superior izquierda se encuentra una flecha, la cuál despliega un menú en la cuál deberá seleccionar
entre 3 bases de datos disponibles; Estas corresponden a una planta distinta y están ordenadas conforme a su número de datos.

**Ojo: Tener en cuenta que mientras más grande sea la base de datos mayor será el tiempo de ejecución.** """)

st.write("""# 2nd Paso: Selección de gases

En el mismo menú del paso anterior se encuentra el apartado "Gases", en donde encontrará checkbox para seleccionar los gases que monitorearemos..

**Ojo: A mayor número de gases mayor será el tiempo de ejecución.**

Una vez seleccionado la base de datos y los gases a analizar, podemos empezar haciendo clic en el 
###### siguiente boton. """)

lista = []
header = ["Date"]
p= df[0]

if C2H2:
    a = df[[0,1]]
    lista.append(a)
    header.append("Acetileno")

if H2:
    b = df[[0,2]]
    lista.append(b)
    header.append("Hidrogeno")

if C2H4:
    c = df[[0,3]]
    lista.append(c)
    header.append("Etileno")

if CO:
    d = df[[0,4]]
    lista.append(d)
    header.append("Monoxido de carbono")

if C2H6:
    e = df[[0,5]]
    lista.append(e)
    header.append("Etano")

if CH4:
    f = df[[0,6]]
    lista.append(f)
    header.append("Metano")

if len(lista)> 0:

    for i in lista:
        p = pd.merge(p,i,on = 0, how='outer')
    q = p.copy()
    q.columns = header
    q["Date"] = pd.to_datetime(q["Date"]).dt.strftime("%Y-%m-%d %H:%M:%S")

if st.button("Comenzar"):
    st.write("""# Modelos de Machine Learning
    
    Para la logra la detección de anomalias en esta app se hace uso de dos modelos de Machine Learning,
    Autoencoder para cuando se desea detectar anomalias en 2 o más gases y 
    Isolation Forest, cuando solamente se analiza un gas.""")
    
    if len(header) > 2:

        st.write("## Autoencoder")
        st.write("""Los autoencoders son redes neuronales artificiales, entrenadas de manera no supervisada,que tienen como objetivo
        aprender primero las representaciones codificadas de nuestros datos y luego generar los datos de entrada
        a partir de las representaciones codificadas aprendidas.""")
    
    else:

        st.write("## Isolation Forest")

        st.write("""En un bosque de aislamiento, los datos se submuestrean aleatoriamente, 
        estos se procesan en una estructura de árbol basada en características 
        seleccionadas aleatoriamente.
        """)
    ######### Reproduccion tiempo real
    db = []
    p.drop([0],inplace=True, axis=1)
    #st.write(p)

    for i, row in p.iterrows():
        db.append(row)

    db = pd.DataFrame(db)

    ############## Grafica
    fig = px.line(db)
    st.write(""" ### Valores de los gases seleccionados a través del tiempo.""")
    st.write(fig)
    

    #### AUTOENCODER
    gases = []
    if len(header) == 2:
        gases = [0]

    elif len(header) == 3:
        gases = [0,1]

    elif len(header) == 4:
        gases = [0,1,2]

    elif len(header) == 5:
        gases = [0,1,2,3]

    elif len(header) == 6:
        gases = [0,1,2,3,4]

    elif len(header) == 7:
        gases = [0,1,2,3,4,5]

    q["Anomalias"]=np.ones(len(p)) ##################

    if len(header) >= 3:

        X_train = db[0:round((len(db)/3)*2)]
        X_test = db[round((len(db)/3)*2):]
        n_features = len(header)-1 #para gases
        y_train = np.zeros(round((len(db)/3)*2))
        y_test = np.zeros(len(db)-round((len(db)/3)*2))
        y_train[round((len(db)/3)*2):] = 1
        y_test[len(db)-round((len(db)/3)*2):] = 1
        y_train = pd.DataFrame(y_train)
        y_test = pd.DataFrame(y_test)

    #estandarizacion
        X_train = StandardScaler().fit_transform(X_train)
        X_train = pd.DataFrame(X_train)
        X_test = StandardScaler().fit_transform(X_test)
        X_test = pd.DataFrame(X_test)

        clf = AutoEncoder(hidden_neurons =[25, 2, 2, 25],contamination=.3)
        clf.fit(X_train)

    # Get the outlier scores for the train data
        y_train_scores = clf.decision_scores_  

    # Predict the anomaly scores
        y_test_scores = clf.decision_function(X_test)  # outlier scores
        y_test_scores = pd.Series(y_test_scores)
        fig3 = plt.figure(figsize=(10,4))   
        plt.hist(y_test_scores, bins='auto')  
        plt.title("Histogram for Model Clf Anomaly Scores")
        plt.show();
        
        
        df_test = X_test.copy()
        df_test['score'] = y_test_scores
        df_test['cluster'] = np.where(df_test['score']<4, 0, 1)
        df_test['cluster'].value_counts()

        t = df_test.groupby('cluster').mean()
        indices = pd.DataFrame(np.where(y_test_scores > (t["score"].min()+(t["score"].max()/2.8))))
        #st.write(indices)
        for i, j  in indices.iteritems():
            q["Anomalias"][j]=-1
        X_test = db[round((len(db)/3)*2):]
        X_test.reset_index(inplace=True)
        X_test.drop(["index"],axis=1,inplace=True)
        fig2 = plt.figure(2)
        plt.plot(X_test.index,X_test.iloc[:, gases])
        plt.vlines([indices],0,X_test.max().max(),"r")
        plt.xlabel('Date Time')
        plt.ylabel('Gases')
        plt.show();
     
        st.write("""### Comportamiento de los gases agrupados según sus valores""")
        st.write(fig3)

        st.write("""Permite definir valores “normales” y valores “anómalos”.""")

        st.write("""### Resultado del modelo""")        
        st.write(fig2)
        
        st.write("""Las lineas verticales (rojas) son las anomalias detectadas por el Autoencoder, 
        estas representan el momento en la que el conjunto de gases salen de su comportamiento “normal”.""")
    
        

    ####### ISOLATION FOREST
    if len(header) == 2:
        CO = db.iloc[:, [0]]

    #Parámetros
        outliers_fraction = float(.1)
        scaler = StandardScaler()
        np_scaled = scaler.fit_transform(CO.values.reshape(-1, 1))
        data = pd.DataFrame(CO.iloc[:, [0]])
        model =  IsolationForest(contamination=outliers_fraction)
        model.fit(data)

        CO['anomaly'] = model.predict(data)
        q["Anomalias"] =model.predict(data)

        if CO.columns[0]==1:
            fig4, ax = plt.subplots(figsize=(10,6))
            a = CO.loc[CO['anomaly'] == -1, [1]] #anomaly
            #st.write(a)
            ax.plot(CO.index, CO.iloc[:, [0]], color='black', label = 'Normal')
            ax.scatter(a.index,a.iloc[:, [0]], color='red', label = 'Anomaly')
            plt.title("Acetileno")
            plt.legend()
            plt.show();
            st.write(fig4,ax)

        elif CO.columns[0]==2:
            fig4, ax = plt.subplots(figsize=(10,6))

            a = CO.loc[CO['anomaly'] == -1, [2]] #anomaly
            #st.write(a)
            ax.plot(CO.index, CO.iloc[:, [0]], color='black', label = 'Normal')
            ax.scatter(a.index,a.iloc[:, [0]], color='red', label = 'Anomaly')
            plt.title("Hidrogeno")
            plt.legend()
            plt.show();
            st.write(fig4,ax)

        elif CO.columns[0]==3:
            fig4, ax = plt.subplots(figsize=(10,6))

            a = CO.loc[CO['anomaly'] == -1, [3]] #anomaly
            #st.write(a)
            ax.plot(CO.index, CO.iloc[:, [0]], color='black', label = 'Normal')
            ax.scatter(a.index,a.iloc[:, [0]], color='red', label = 'Anomaly')
            plt.title("Etileno")
            plt.legend()
            plt.show();
            st.write(fig4,ax)

        elif CO.columns[0]==4:
            fig4, ax = plt.subplots(figsize=(10,6))

            a = CO.loc[CO['anomaly'] == -1, [4]] #anomaly
            #st.write(a)
            ax.plot(CO.index, CO.iloc[:, [0]], color='black', label = 'Normal')
            ax.scatter(a.index,a.iloc[:, [0]], color='red', label = 'Anomaly')
            plt.title("Monoxido de Carbono")
            plt.legend()
            plt.show();
            st.write(fig4,ax)
            
        elif CO.columns[0]==5:
            fig4, ax = plt.subplots(figsize=(10,6))

            a = CO.loc[CO['anomaly'] == -1, [5]] #anomaly
            #st.write(a)
            ax.plot(CO.index, CO.iloc[:, [0]], color='black', label = 'Normal')
            ax.scatter(a.index,a.iloc[:, [0]], color='red', label = 'Anomaly')
            plt.title("Etano")
            plt.legend()
            plt.show();
            st.write(fig4,ax)

        elif CO.columns[0]==6:
            fig4, ax = plt.subplots(figsize=(10,6))

            a = CO.loc[CO['anomaly'] == -1, [6]] #anomaly
            #st.write(a)
            ax.plot(CO.index, CO.iloc[:, [0]], color='black', label = 'Normal')
            ax.scatter(a.index,a.iloc[:, [0]], color='red', label = 'Anomaly')
            plt.title("Metano")
            plt.legend()
            plt.show();
            st.write(fig4,ax)   
    
    # visualization

    header.append("Anomalias")
    q["Anomalias"]=q["Anomalias"].replace(1, 0)
    q["Anomalias"]=q["Anomalias"].replace(-1, q[header[1:]].max().max())
    gs = []
    dat = []
    va = []
    co = 0
    c = ["#FEAF3E","#FBFE1D","#54FE1D","#1DBAFE","#0E5C7E","#885693","#E52323"]
    gr = pd.DataFrame()
    for i in range(len(q)):
        for k in range(len(header[1:])):
            dat.append(q["Date"][i])
        for k in header[1:]:
            va.append(q[k][i])
        for k in header[1:]:
            gs.append(k)
    #colors= c[:len(header)]
    #colors.append("#E52323")
    colores={}
    for i in header[1:]:
        colores[i]=c[co]
        co += 1
    colores["Anomalias"] = "#E52323" 
    gr["Date"]=dat
    gr["Gas"]= gs
    gr["Valor"]= va
    st.write("# Simulación de tiempo real")
    st.write("""Las fechas de las anomalias son:""")
    st.write(q["Date"].loc[(q['Anomalias']> 0)])
    if st.button("Simulación tiempo real"):
        fig = px.bar(gr, x= "Gas", y= "Valor",color="Gas", 
        color_discrete_map=colores, animation_frame= "Date", 
        animation_group= "Gas")
        fig.update_layout(width=800)
        fig.update_yaxes(range=[0,(gr["Valor"].max().max())//3])
        st.write(fig)
        st.write("""Simulación de la recabación de los datos a traves de un periodo de tiempo.
        El valor asignado a la anomalia es simbolico y solo sirve como representación de su existencia.""")

st.write(""" ### Visualizador de variables seleccionadas""")

if st.checkbox('Visualizar base de datos seleccionada'):
    headz = ["Date","Acetileno","Hidrogeno","Etileno","Monoxido de carbono","Etano","Metano"]
    r = df.copy()
    r.columns = headz
    st.write(r)

if len(lista)> 0:
    if st.checkbox('Visualizar variables seleccionadas'):
        st.write(q.drop(columns=['Date']))


