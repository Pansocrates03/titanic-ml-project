# streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def show_tablaResumen():
    try:
        df = pd.read_csv('data/processed/titanic_dataset_features.csv')
    except FileNotFoundError:
        st.error("El archivo 'titanic_dataset_features.csv' no se encontró en la ruta 'data/processed/'.")
        return
    except Exception as e:
        st.error(f"Ocurrió un error al cargar el archivo CSV: {e}")
        return
    st.dataframe(df.describe().T)

    

def show_correlacionDeSupervivenciaInteractiva(group_by_column = 'Pclass'):
    
    # Normalización del input
    possibleGroups = ['Pclass', 'Sex', 'Embarked', 'AgeGroup', 'FamilySize']
    if group_by_column not in possibleGroups:
        group_by_column = 'Pclass'
    title = f'Tasa de Supervivencia por {group_by_column}'

    # Visualizacion 1
    df = pd.read_csv('data/processed/titanic_dataset_features.csv')

    # Calcular la tasa de supervivencia por clase y sexo
    survival_rate = df.groupby([group_by_column])

    # Crear un gráfico de barras
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x=group_by_column, y='Survived', palette='viridis')
    plt.title(title)
    plt.xlabel(group_by_column) 
    plt.ylabel('Tasa de Supervivencia')

    # Mostrar la gráfica
    st.pyplot(plt)
    plt.clf()  # Limpia la figura para evitar superposiciones en futuras gráficas

def show_matrizDeCorrelacion():
    df = pd.read_csv('data/processed/titanic_dataset_features.csv')
    corr = df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title('Matriz de Correlación')
    st.pyplot(plt)
    plt.clf()  # Limpia la figura para evitar superposiciones en futuras gráficas

st.set_page_config(
    page_title="Exploración",
    page_icon="🌎",
)


st.title('Exploración de Datos')
show_tablaResumen()

st.subheader('Visualizaciones interactivas del EDA')
#show_matrizDeCorrelacion()

st.subheader('Filtros por características demográficas')
options = ["Supervivencia por lugar de embarque", "Supervivencia por clase", "Supervivencia por sexo", "Supervivencia por edad", "Supervivencia por tamaños de familia"]
selected_option = st.selectbox("Selecciona una visualización:", options)

if selected_option == "Supervivencia por lugar de embarque":
    show_correlacionDeSupervivenciaInteractiva("Embarked")
elif selected_option == "Supervivencia por clase":
    show_correlacionDeSupervivenciaInteractiva("Pclass")
elif selected_option == "Supervivencia por sexo":
    show_correlacionDeSupervivenciaInteractiva("Sex")
elif selected_option == "Supervivencia por edad":
    show_correlacionDeSupervivenciaInteractiva("AgeGroup")
elif selected_option == "Supervivencia por tamaños de familia":
    show_correlacionDeSupervivenciaInteractiva("FamilySize")



st.subheader('Estadísticas dinámicas')


