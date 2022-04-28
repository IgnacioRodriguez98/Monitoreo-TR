import plotly.express as px
import streamlit as st
import pandas as pd
import time
import numpy as np
from github import Github
from github import InputGitTreeElement
from datetime import datetime



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

p.to_csv("prueba.csv")

#l= pd.DataFrame()
#k= 0
#for i in p:
#    l= l.append([p[i]])
#    st.write(l)
#    time.sleep(1)

#######################Prueba github ####################
if st.button('Cargar a Github'):
    #create test pd df to upload
    d = {'col1': [1, 2, 3], 'col2': [4, 5, 6]}
    df = pd.DataFrame(d)
    #convert pd.df to text. This avoids writing the file as csv to local and again reading it
    df2 = df.to_csv(sep=',', index=False)

    #list files to upload and desired file names with which you want to save on GitHub
    file_list = [df2,df2]
    file_names = ['Test.csv','Test2.csv']

    #Specify commit message
    commit_message = 'Test Python'

    #Create connection with GiHub
    user = "IgnacioRodriguez98"
    password = "sespio007"
    g = Github(user,password)

    #Get list of repos
    for repo in g.get_user().get_repos():
        print(repo.name)
        repo.edit(has_wiki=False)

    #Create connection with desired repo
    repo = g.get_user().get_repo('Monitoreo-TR')

    #Check files under the selected repo
    x = repo.get_contents("")
    for labels in x:
        print(labels)
    x = repo.get_contents("Test.csv") #read a specific file from your repo

    #Get available branches in your repo
    x = repo.get_git_refs()
    for y in x:
        print(y)
    # output eg:- GitRef(ref="refs/heads/master")

    #Select required branch where you want to upload your file.
    master_ref = repo.get_git_ref("master")

    #Finally, putting everything in a function to make it re-usable

    def updategitfiles(file_names,file_list,userid,pwd,Repo,branch,commit_message =""):
        if commit_message == "":
        commit_message = "Data Updated - "+ datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        g = Github(userid,pwd)
        repo = g.get_user().get_repo(Repo)
        master_ref = repo.get_git_ref("heads/"+branch)
        master_sha = master_ref.object.sha
        base_tree = repo.get_git_tree(master_sha)
        element_list = list()
        for i in range(0,len(file_list)):
            element = InputGitTreeElement(file_names[i], '100644', 'blob', file_list[i])
            element_list.append(element)
        tree = repo.create_git_tree(element_list, base_tree)
        parent = repo.get_git_commit(master_sha)
        commit = repo.create_git_commit(commit_message, tree, [parent])
        master_ref.edit(commit.sha)
        print('Update complete')

    updategitfiles(file_names,file_list,user,password,'Monitoreo-TR','master')
