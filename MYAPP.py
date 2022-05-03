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
     ('1', '2', '3'))

    st.write("# Selecciona los gases a analizar:")
    C2H2 = st.checkbox("Acetileno")
    H2 = st.checkbox("Hidrógeno")
    C2H4 = st.checkbox("Etileno")
    CO = st.checkbox("Monóxido de carbono")
    C2H6 = st.checkbox("Etano")
    CH4 = st.checkbox("Metano")

if database == '1':
        st.write("""# Has seleccionado la base de datos 1""")
        df = pd.read_csv('https://raw.githubusercontent.com/IgnacioRodriguez98/Monitoreo-TR/main/Data/norm.csv', header=None)
        st.write(df)

elif database == '2':
    st.write("""# Has seleccionado la base de datos 2""")
    df = pd.read_csv('https://raw.githubusercontent.com/IgnacioRodriguez98/Monitoreo-TR/main/Data/normCH.csv', header=None)
    st.write(df)

else:
    st.write("""# Has seleccionado la base de datos 3""")
    df = pd.read_csv('https://raw.githubusercontent.com/IgnacioRodriguez98/Monitoreo-TR/main/Data/normJA.csv', header=None)
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
    p.drop([0],inplace=True, axis=1)

if C2H2== False |H2 == False | C2H4 == False |CO == False |C2H6 ==False |CH4 == False:
   p="""### No hay gases seleccionados, por favor selecciona al menos uno"""

st.write(p)

if st.button("Apriete aqui"):
    x = [2016,2017,2018,2019,2020,2021]
    y = [45,46,48,50,51]
    w = []
    z = []
    for i in range(6):
        w.append(x[i])
        z.append(y[i])
        plt.plot(w,z)
        time.sleep(2)

#l= pd.DataFrame()
#k= 0
#for i in p:
#    l= l.append([p[i]])
#    st.write(l)
#    time.sleep(1)

#######################Prueba github ####################
if st.button('Cargar a Github'):
    g = Github("IgnacioRodriguez98", "password")
    # Upload to github
    git_prefix = '/Monitoreo-TR/Data'
    git_file = git_prefix + p.to_csv("prueba.csv")
    if git_file in all_files:
        contents = repo.get_contents(git_file)
        repo.update_file(contents.path, "committing files", content, contents.sha, branch="master")
        print(git_file + ' UPDATED')
    else:
        repo.create_file(git_file, "committing files", content, branch="master")
        print(git_file + ' CREATED')

if st.button('Prueba de Grafica'):
    def func(t, line):
        t = np.arange(0,t,0.1)
        y = np.sin(t)
        line.set_data(t, y)
        return line
    
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 100), ylim=(-1.2, 1.22))
    redDots = plt.plot([], [], 'ro')
    line = plt.plot([], [], lw=2)
    

    # Creating the Animation object
    line_ani = animation.FuncAnimation(fig, func, frames=np.arange(1,100,0.1), fargs=(line), interval=100, blit=False)
    #line_ani.save(r'Animation.mp4')
    
    

    #HtmlFile = line_ani.to_html5_video()
    #with open("myvideo.html","w") as f:
    #    print(line_ani.to_html5_video(), file=f)
        
    #HtmlFile = open("myvideo.html", "r")
    #HtmlFile="myvideo.html"
    #source_code = HtmlFile.read() 
    #components.html(source_code, height = 900,width=900)