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
   # b = p.drop([0],inplace=True, axis=1)

if C2H2== False |H2 == False | C2H4 == False |CO == False |C2H6 ==False |CH4 == False:
   p="""### No hay gases seleccionados, por favor selecciona al menos uno"""

#st.write(p)
#st.write(p.iloc[:,1:])

q = p.copy()
#headers = header[0:len(lista)]
q.columns = header
q["Date"] = pd.to_datetime(q["Date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
#st.write(header[1:])
#e = q.drop(["DA"],inplace=True, axis=1)
gs = []
dat = []
va = []
nm = []
c = 0
gr = pd.DataFrame()
for i in range(len(q)):
    for k in range(len(lista)+1):
        dat.append(q["Date"][i])
    for k in lista:
        va.append(q[k][i])
    #dat.append(q["Date"][i])
    #dat.append(q["Date"][i])
    #dat.append(q["Date"][i])
    #dat.append(q["Date"][i])
    #dat.append(q["Date"][i])
    #dat.append(q["Date"][i])
    #va.append(q["Acetileno"][i])
    #va.append(q["Hidrogeno"][i])
    #va.append(q["Etileno"][i])
    #va.append(q["Monoxido de carbono"][i])
    #va.append(q["Etano"][i])
    #va.append(q["Metano"][i])
    gs.append(lista)
    #gs.append("Anomalia")

#for i in range(len(gs)):
#    nm.append(i)

#gr["Num"]= nm
gr["Date"]=dat
gr["Gas"]= gs
gr["Valor"]= va


st.write(q)
st.write(gr)
if st.button("Simulación tiempo real"):
    #st.write(len(q))
    fig = px.bar(gr, x= "Gas", y= "Valor", color="Gas",
    animation_frame= "Date", 
    animation_group= "Gas")
    fig.update_layout(width=800)
    st.write(fig)

if st.button("Simulación tiempo real 2"):
    #st.write(len(q))
    fig = px.bar(q, x= "Gas", y= "Valor", color="Gas",
    animation_frame= "Date", 
    animation_group= "Gas")
    fig.update_layout(width=800)
    st.write(fig)

if st.button("Matplotlib animation"):
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots()
    ax.set(xlim=(-3, 3), ylim=(-1, 1))
    x = np.linspace(-3, 3, 91)
    t = np.linspace(1, 25, 30)
    X2, T2 = np.meshgrid(x, t)
    sinT2 = np.sin(2 * np.pi * T2 / T2.max())
    F = 0.9 * sinT2 * np.sinc(X2 * (1 + sinT2))
    line, = ax.plot(x, F[0, :], color='k', lw=2)
    def animate(i):
        line.set_ydata(F[i, :])
    anim = animation.FuncAnimation(fig, animate, interval=100, frames=len(t) - 1)
    anim.save('503.gif')
    st.write(plt)