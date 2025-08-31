import streamlit as st

# Variables
var1 = "valor1"

def click():
    var1 = "valor2"

st.title("Predicción")
st.write("¿Sobrevivirías al Titanic? Usa el modelo para averiguarlo.")

inputData = {
    "name": "John Doe",
    "age": 30,
    "family": 0,
    "clase": "Tercera",
    "sex": "Masculino"
}

# Input data
inputData["name"] = st.text_input("Nombre")
inputData["age"] = st.number_input("Edad", min_value=0, max_value=100, value=30)
inputData["family"] = st.number_input("Familiares a bordo", min_value=0, max_value=10, value=0)
inputData["clase"] = st.selectbox("Clase", ["Primera", "Segunda", "Tercera"])
inputData["sex"] = sex = st.selectbox("Sexo", ["Masculino", "Femenino"])
boton = st.button("Predecir Supervivencia", key="predict_button", on_click=click())



st.subheader("Predicción en tiempo real con múltiples modelos")

st.write(var1)

st.write("La posibilidad de supervivencia es de 60%")
st.write(inputData["name"])

st.subheader("Explicación de la predicción (SHAP/LIME)")
st.write("La razón por la que tu posibilidad de supervivencia es del 60% es porque eres mujer y tienes 30 años.")

st.subheader("Intervalos de confianza")