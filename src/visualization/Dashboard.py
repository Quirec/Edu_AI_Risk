import pandas as pd
import streamlit as st

st.title("Edu AI Risk Dashboard")

df = pd.read_csv("data/raw/student-mat.csv")

st.write("Vista previa de los datos")
st.dataframe(df.head())

st.write("Dimensiones del dataset")
st.write(df.shape)

st.write("Resumen estadístico")
st.dataframe(df.describe())

st.write("Columnas")
st.write(df.columns.tolist())