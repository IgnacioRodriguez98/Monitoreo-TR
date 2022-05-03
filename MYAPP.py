import plotly.express as px
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import time
import numpy as np
from github import Github
import streamlit.components.v1 as components
import matplotlib.animation as animation


with st.sidebar:
    option = st.selectbox(
     "Lenguage / Idioma" ,
     ('English', 'Español'))
##########################################################

st.write(option)
##########################################################
with st.sidebar:
    st.write("# Variables")
    database = st.radio(
     "Base de datos",
     ('1 (2432 datos)', '2 (9277 datos)', '3 (21533 datos)'))

    st.write("# Selecciona los gases a analizar:")
    C2H2 = st.checkbox("Acetileno")
    H2 = st.checkbox("Hidrógeno")
    C2H4 = st.checkbox("Etileno")
    CO = st.checkbox("Monóxido de carbono")
    C2H6 = st.checkbox("Etano")
    CH4 = st.checkbox("Metano")

if database == '1 (2432 datos)':
        st.write("""# Has seleccionado la base de datos 1""")
        df = pd.read_csv('https://raw.githubusercontent.com/IgnacioRodriguez98/Monitoreo-TR/main/Data/planta2.csv', header=None)
        st.write(df)

elif database == '2 (9277 datos)':
    st.write("""# Has seleccionado la base de datos 2""")
    df = pd.read_csv('https://raw.githubusercontent.com/IgnacioRodriguez98/Monitoreo-TR/main/Data/planta3.csv', header=None)
    st.write(df)

else:
    st.write("""# Has seleccionado la base de datos 3""")
    df = pd.read_csv('https://raw.githubusercontent.com/IgnacioRodriguez98/Monitoreo-TR/main/Data/planta1.csv', header=None)
    st.write(df)
##################### SELECCION DE GASES ######################

#with st.sidebar:
#   st.write("# Selecciona el tamaño de la ventana:")
#    vent= st.slider("",1,len(df))

lista = []
p= df[0]

if C2H2:
    a = df[[0,1]]
    lista.append(a)

if H2:
    b = df[[0,2]]
    lista.append(b)

if C2H4:
    c = df[[0,3]]
    lista.append(c)

if CO:
    d = df[[0,4]]
    lista.append(d)

if C2H6:
    e = df[[0,5]]
    lista.append(e)

if CH4:
    f = df[[0,6]]
    lista.append(f)

if len(lista)> 0:

    for i in lista:
        p = pd.merge(p,i,on = 0, how='outer')
   # b = p.drop([0],inplace=True, axis=1)

if C2H2== False |H2 == False | C2H4 == False |CO == False |C2H6 ==False |CH4 == False:
   p="""### No hay gases seleccionados, por favor selecciona al menos uno"""

#st.write(p)
st.write(p.iloc[:,1:])

fig = px.line(p, x= 0, y= p.iloc[:,1:],
animation_frame= p.iloc[:, [0]], 
animation_group= p.iloc[:,1:], 
range_y=a.all())
#fig.show()