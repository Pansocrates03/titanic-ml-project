from matplotlib import pyplot as plt
import streamlit as st
import joblib
import numpy as np
import os
import pandas as pd
import shap
import pkg_resources
import xgboost as xgb


# -----------------------------
# 1. Funciones auxiliares
# -----------------------------
def bootstrap_ci(model, X, n_bootstrap=1000, alpha=0.05):
    """Calcula intervalos de confianza con bootstrap para predict_proba."""
    if len(X) == 1:
        prob = model.predict_proba(X)[0][1]
        return [prob], [prob]

    preds = []
    n = len(X)
    for _ in range(n_bootstrap):
        sample_idx = np.random.choice(range(n), size=n, replace=True)
        X_boot = X.iloc[sample_idx]
        preds.append(model.predict_proba(X_boot)[:, 1])
    preds = np.array(preds)
    lower = np.percentile(preds, 100 * alpha / 2, axis=0)
    upper = np.percentile(preds, 100 * (1 - alpha / 2), axis=0)
    return lower, upper


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


MODEL_FILES = {
    "Logistic Regression": "logistic_regression_final.pkl",
    "SVM": "svm_final.pkl",
    "Random Forest": "random_forest_final.pkl",
    "XGBoost": "xgboost_final.pkl",
}

# -----------------------------
# 2. Feature Engineering
# -----------------------------
def build_features(input_data):
    """Construye features a partir del input del usuario"""
    features = {}

    # Datos básicos
    features["Age"] = input_data.get("age", 30)
    features["SibSp"] = input_data.get("SibSp", 0)
    features["Parch"] = input_data.get("parch", 0)
    features["FamilySize"] = features["SibSp"] + features["Parch"] + 1
    features["IsAlone"] = 1 if features["FamilySize"] == 1 else 0

    # Sexo
    sex = input_data.get("sex", "Masculino")
    features["Sex_female"] = 1 if sex=="Femenino" else 0
    features["Sex_male"] = 1 if sex=="Masculino" else 0

    # Clase
    clase = input_data.get("clase", "Tercera")
    features["Pclass_1"] = 1 if clase=="Primera" else 0
    features["Pclass_2"] = 1 if clase=="Segunda" else 0
    features["Pclass_3"] = 1 if clase=="Tercera" else 0

    # Títulos
    name = input_data.get("name","")
    titles = ["Mr","Mrs","Miss","Master","Ms","Dr","Rev","Col","Major","Capt","Sir","Lady","Jonkheer","Dona","Countess"]
    for title in titles:
        features[f"Title_{title}"] = 1 if title in name else 0

    # FarePerPerson (ejemplo simplificado)
    fare_map = {"Primera": 52, "Segunda": 12, "Tercera": 8}
    features["FarePerPerson"] = fare_map.get(clase, 8)

    # Age bins
    age = features["Age"]
    features["AgeGroup_Child"] = 1 if age < 10 else 0
    features["AgeGroup_Teen"] = 1 if 10 <= age < 18 else 0
    features["AgeGroup_Adult"] = 1 if 18 <= age < 60 else 0
    features["AgeGroup_Senior"] = 1 if age >= 60 else 0

    return features


def build_full_features(input_data, model_columns):
    features = build_features(input_data)
    df = pd.DataFrame([features])

    # ⚡ Reindexar según columnas del modelo para evitar errores
    df = df.reindex(columns=model_columns, fill_value=0)

    return df


# -----------------------------
# 3. Streamlit App
# -----------------------------
st.set_page_config()
st.title("🚢 Predicción de Supervivencia en el Titanic")

selected_model_name = st.selectbox("Selecciona el modelo:", list(MODEL_FILES.keys()))
model, model_columns = load_model(MODEL_FILES[selected_model_name])
st.success(f"✅ Modelo cargado: {selected_model_name}")

st.write("### Ingresa los datos del pasajero:")
inputData = {
    "name": st.text_input("Nombre", "John Doe"),
    "age": st.number_input("Edad", min_value=0, max_value=100, value=30),
    "parch": st.number_input("Padres o hijos a bordo", min_value=0, max_value=10, value=0),
    "SibSp": st.number_input("Hermanos a bordo", min_value=0, max_value=10, value=0),
    "clase": st.selectbox("Clase", ["Primera", "Segunda", "Tercera"]),
    "sex": st.selectbox("Sexo", ["Masculino", "Femenino"])
}

if st.button("Predecir Supervivencia"):
    X_user = build_full_features(inputData, model_columns=model_columns)

    prob = model.predict_proba(X_user)[0][1]
    st.success(f"Probabilidad de supervivencia: {prob:.2%}")

    try:
        lower, upper = bootstrap_ci(model, X_user)
        st.subheader("Intervalos de confianza")
        st.write(f"95% CI: [{lower[0]:.2%}, {upper[0]:.2%}]")
    except Exception as e:
        st.warning(f"No se pudo calcular el intervalo de confianza: {e}")

    # SHAP
    st.subheader("Explicación de la predicción (SHAP)")
    try:
        # Crea el explainer y los shap_values como ya lo haces
        if "XGB" in selected_model_name or "Random Forest" in selected_model_name:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, X_user, feature_perturbation="interventional")

        shap_values = explainer.shap_values(X_user)
        shap.initjs()

        # 1. Generar el force_plot
        shap.force_plot(explainer.expected_value, shap_values[0], X_user.iloc[0,:], matplotlib=True, show=False)
        st.pyplot(bbox_inches='tight', dpi=150, pad_inches=0.1)
        
        # 2. Generar el summary_plot
        # Cierra la figura anterior para evitar que se mezclen los gráficos
        plt.clf() 
        plt.cla()
        plt.close('all')

        # Llama a shap.summary_plot y dile que dibuje en los ejes `ax`
        # El argumento `show=False` evita que matplotlib muestre el gráfico directamente
        # Nota: He eliminado el argumento `ax=ax` para compatibilidad con versiones antiguas
        shap.summary_plot(shap_values, X_user, plot_type="bar", show=False)

        # st.pyplot() ya usa la figura actual, pero es bueno ser explícito
        st.pyplot()

    except Exception as e:
        st.warning(f"No se pudo generar SHAP: {e}")

else:
    st.subheader("Intervalos de confianza")
    st.write("👉 Ingresa datos y haz clic en 'Predecir Supervivencia'.")
