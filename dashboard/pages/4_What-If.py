import streamlit as st
import pandas as pd

import shap
from matplotlib import pyplot as plt
import joblib
import os

def load_model(model_name):
    """Carga un modelo desde la carpeta models y obtiene las columnas esperadas."""
    pages_dir = os.path.dirname(__file__)
    dashboard_dir = os.path.dirname(pages_dir)
    project_dir = os.path.dirname(dashboard_dir)
    model_path = os.path.join(project_dir, "models", model_name)
    model = joblib.load(model_path)

    # Intentar obtener las columnas usadas en entrenamiento
    model_columns = getattr(model, "feature_names_in_", None)

    # Si no existen, puedes definirlas manualmente según tu entrenamiento
    if model_columns is None:
        # Lista de las 30 columnas originales del Logistic Regression
        model_columns = [
            "Age","SibSp","Parch","FamilySize","IsAlone","FarePerPerson",
            "Sex_female","Sex_male",
            "Pclass_1","Pclass_2","Pclass_3",
            "Title_Mr","Title_Mrs","Title_Miss","Title_Master","Title_Ms",
            "Title_Dr","Title_Rev","Title_Col","Title_Major","Title_Capt",
            "Title_Sir","Title_Lady","Title_Jonkheer","Title_Dona","Title_Countess",
            "AgeGroup_Child","AgeGroup_Teen","AgeGroup_Adult","AgeGroup_Senior"
        ]
    return model, model_columns

model, model_columns = load_model("logistic_regression_final.pkl")

# -----------------------------
# 2. Sección de "What-If"
# -----------------------------
st.title("Sección de 'What-If'")
st.subheader("Herramienta de análisis contrafactual")

st.markdown("Ajusta las características del pasajero para ver cómo cambian la predicción y las explicaciones del modelo.")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Sliders para modificar características")
    age = st.slider("Edad", min_value=0, max_value=100, value=30)
    sex = st.selectbox("Sexo", ["Masculino", "Femenino"])
    pclass = st.selectbox("Clase", ["Primera", "Segunda", "Tercera"])

with col2:
    # Construir las características basadas en los inputs
    features = {
        "Age": age,
        "SibSp": 0,
        "Parch": 0,
        "FamilySize": 1,
        "IsAlone": 1,
        "FarePerPerson": 10, # Valor de ejemplo
        "Sex_female": 1 if sex == "Femenino" else 0,
        "Sex_male": 1 if sex == "Masculino" else 0,
        "Pclass_1": 1 if pclass == "Primera" else 0,
        "Pclass_2": 1 if pclass == "Segunda" else 0,
        "Pclass_3": 1 if pclass == "Tercera" else 0,
        "Title_Mr": 1, "Title_Mrs": 0, "Title_Miss": 0, "Title_Master": 0, "Title_Ms": 0,
        "Title_Dr": 0, "Title_Rev": 0, "Title_Col": 0, "Title_Major": 0, "Title_Capt": 0,
        "Title_Sir": 0, "Title_Lady": 0, "Title_Jonkheer": 0, "Title_Dona": 0, "Title_Countess": 0,
        "AgeGroup_Child": 1 if age < 10 else 0,
        "AgeGroup_Teen": 1 if 10 <= age < 18 else 0,
        "AgeGroup_Adult": 1 if 18 <= age < 60 else 0,
        "AgeGroup_Senior": 1 if age >= 60 else 0,
    }
    
    X_whatif = pd.DataFrame([features]).reindex(columns=model_columns, fill_value=0)

    st.subheader("Visualización del cambio en probabilidad")
    prob = model.predict_proba(X_whatif)[0][1]
    st.metric(label="Probabilidad de Supervivencia", value=f"{prob:.2%}")


