import streamlit as st
import pandas as pd

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

a = df[1]
b = df[2]
c = pd.merge(a,b, how='outer')
if C2H2 == True and (H2,C2H4,CO,C2H6,CH4) == False:
    database2 = database2.end(df[1]) 

elif C2H2 == True and H2 == True:
    database2 = df[[1,2]]
 
if H2 == True:
    database = database["2"]

if C2H4 == True:
    database = database["3"]

if CO == True:
    database = database["4"] 

if C2H6 == True:
    database = database["5"]

if CH4 == True:
    database = database["6"]

st.write(c)
