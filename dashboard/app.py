# streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np

st.title('Sección de Exploración')

st.subheader('Visualizaciones interactivas del EDA')
st.subheader('Filtros por características demográficas')
st.subheader('Estadísticas dinámicas')

st.title("Sección de Predicción")

st.subheader("Interfaz para ingresar datos de un pasajero")
st.subheader("Predicción en tiempo real con múltiples modelos")
st.subheader("Explicación de la predicción (SHAP/LIME)")
st.subheader("Intervalos de confianza")

st.title("Sección de Análisis de Modelos")

st.subheader("Comparación de métricas entre modelos")
st.subheader("Visualización de feature importance")
st.subheader("Análisis de errores interactivo")

st.title("Sección de 'What-If'")

st.subheader("Herramienta de análisis contrafactual")
st.subheader("Sliders para modificar características")
st.subheader("Visualización del cambio en probabilidad")