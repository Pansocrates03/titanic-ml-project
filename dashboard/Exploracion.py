# streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Exploración",
    page_icon="🌎",
)

st.set_page_config(page_title="Plotting Demo", page_icon="📈")

st.subheader('Visualizaciones interactivas del EDA')

# Opciones de visualización
options = ["Supervivencia por tamaños de familia", "Supervivencia por clase y género"]
visualizaciones = st.selectbox("Sexo", options)

# Visualizacion 1
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../data/processed/titanic_dataset_features.csv')

df['TamañoFamilia'] = df['SibSp'] + df['Parch'] + 1  # Tamaño de la familia dentro del barco
df["OrigenClase3"] = (df['Pclass'] == 3).astype(int) # Creación de variable binaria para una mejor interpretación
df["OrigenEmbarked_Southampton"] = (df['Embarked'] == 'S').astype(int)
subgrupo1 = (df["OrigenClase3"] == 1) & (df["OrigenEmbarked_Southampton"] == 1) # Creación del subgrupo que respeta las combinanciones anteriores
df['Categorias_Familia'] = pd.cut(df['TamañoFamilia'], bins=[0,1,3,6,100], labels=['1','2-3','4-6','7+']) # Categorías de familia con etiquetas
print("Total de Pasajeros de Tercera & Southampton por Familia :", df[subgrupo1].shape[0])
print("Distribución (Tercera Clase & Southampton por Tamaño de Familia):")
print(df[subgrupo1]['Categorias_Familia'].value_counts(dropna=False))

df[subgrupo1].groupby('Categorias_Familia')['Survived'].agg(['count','mean']).rename(columns={'mean':'Supervivencia'})

# Se calcula la supervivencia por el subgrupo de categorias de tamaños de familia
survival_rate_subgrupo1 = df[subgrupo1].groupby('Categorias_Familia')['Survived'].mean().reset_index()

# Se crea un boxplot
plt.figure(figsize=(8, 5))
sns.barplot(data=survival_rate_subgrupo1, x='Categorias_Familia', y='Survived', palette='viridis')

# Add titles and labels
plt.title('Supervivencia por Categorias de Tamaños de Familia (Tercera Clase & Southampton)')
plt.xlabel('Categorias de Tamaños de Familia')
plt.ylabel('Supervivencia')

# Desplegar gráfica
plt.show()

# Desplegar gráfica en Streamlit
st.pyplot(plt)
plt.clf()  # Limpia la figura para evitar superposiciones en futuras gráficas



st.subheader('Filtros por características demográficas')
st.subheader('Estadísticas dinámicas')


