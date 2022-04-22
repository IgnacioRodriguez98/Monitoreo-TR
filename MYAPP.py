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


if C2H2:
    a = df[[0,1]]
    lst.append(a)

elif H2:
    b = df[[0,2]]
    lst.append(b)

elif C2H4:
    c = df[[0,3]]
    lst.append(c)

elif CO:
    d = df[[0,4]]
    lst.append(d)

elif C2H6:
    e = df[[0,5]]
    lst.append(e)

elif CH4:
    f = df[[0,6]]
    lst.append(f)

else:
    p="""### No hay gases seleccionados, por favor selecciona al menos uno"""

p= df[0]

if  len(lst) > 0:
    for i in lst:
        p = pd.merge(p,i,on = 0, how='outer')
    p.drop([0],inplace=True, axis=1)

st.write(p)
