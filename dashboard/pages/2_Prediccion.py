from matplotlib import pyplot as plt
import streamlit as st
import joblib
import numpy as np
import os
import pandas as pd
import shap
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# 1. Funciones auxiliares

def bootstrap_ci(model, X, n_bootstrap=1000, alpha=0.05):
    """
    Calculamos los intervalos de confianza (CI) para la probabilidad de supervivencia
    usando el método bootstrap.
    
    - model: modelo entrenado (requerimos de predict_proba)
    - X: es el dataframe con los features de entrada (30 columnas)
    - n_bootstrap: número de muestras bootstrap
    - alpha: nivel de significancia (0.05 => 95% CI)
    
    Regresa:
    - lower, upper: límites inferior y superior del intervalo
    """
    # Caso especial: si solo hay 1 muestra
    if len(X) == 1:
        prob = model.predict_proba(X)[0][1]  # probabilidad de clase 1
        return [prob], [prob]

    # Caso general: bootstrap con múltiples muestras
    preds = []
    n = len(X)
    for _ in range(n_bootstrap):
        # Selecciona indices aleatorios con reemplazo
        sample_idx = np.random.choice(range(n), size=n, replace=True)
        X_boot = X.iloc[sample_idx]
        preds.append(model.predict_proba(X_boot)[:, 1])  # probabilidad clase 1
    
    preds = np.array(preds)
    # Calculamos percentiles para el intervalo de confianza
    lower = np.percentile(preds, 100 * alpha / 2, axis=0)
    upper = np.percentile(preds, 100 * (1 - alpha / 2), axis=0)
    return lower, upper


def load_model(model_name):
    """
    Cargamos un modelo entrenado desde la carpeta "models" y obtenemos
    las columnas esperadas durante el entrenamiento.
    
    - model_name: nombre del archivo .pkl del modelo
    """
    # Construimos rutas de manera relativa
    pages_dir = os.path.dirname(__file__)
    dashboard_dir = os.path.dirname(pages_dir)
    project_dir = os.path.dirname(dashboard_dir)
    model_path = os.path.join(project_dir, "models", model_name)
    
    # Cargamos el modelo serializado
    model = joblib.load(model_path)

    # Intentamos obtener las columnas usadas en el entrenamiento
    model_columns = getattr(model, "feature_names_in_", None)

    # Si no existen, las definimos manualmente (lista de 30 columnas)
    if model_columns is None:
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


# Diccionario de archivos de modelos
MODEL_FILES = {
    "Logistic Regression": "logistic_regression_final.pkl",
    "SVM": "svm_final.pkl",
    "Random Forest": "random_forest_final.pkl",
    "XGBoost": "xgboost_final.pkl",
}


# 2. Feature Engineering

def build_features(input_data):
    """
    Construimos el diccionario de features a partir de los datos ingresados
    por el usuario en la interfaz de Streamlit.
    """
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

    # Títulos en el nombre
    name = input_data.get("name","")
    titles = ["Mr","Mrs","Miss","Master","Ms","Dr","Rev","Col","Major","Capt","Sir","Lady","Jonkheer","Dona","Countess"]
    for title in titles:
        features[f"Title_{title}"] = 1 if title in name else 0

    # Tarifa promedio por persona según clase
    fare_map = {"Primera": 52, "Segunda": 12, "Tercera": 8}
    features["FarePerPerson"] = fare_map.get(clase, 8)

    # Rango de edad
    age = features["Age"]
    features["AgeGroup_Child"] = 1 if age < 10 else 0
    features["AgeGroup_Teen"] = 1 if 10 <= age < 18 else 0
    features["AgeGroup_Adult"] = 1 if 18 <= age < 60 else 0
    features["AgeGroup_Senior"] = 1 if age >= 60 else 0

    return features


def build_full_features(input_data, model_columns):
    """
    Construimos un dataframe con todas las columnas requeridas por el modelo,
    rellenando con 0 cualquier columna faltante.
    """
    features = build_features(input_data)
    df = pd.DataFrame([features])
    df = df.reindex(columns=model_columns, fill_value=0)
    return df



# 3. Streamlit App
st.set_page_config(page_title="Predicción Titanic")
st.title("🚢 Predicción de Supervivencia en el Titanic")

# Selector de modelo
selected_model_name = st.selectbox("Selecciona el modelo:", list(MODEL_FILES.keys()))
model, model_columns = load_model(MODEL_FILES[selected_model_name])
st.success(f"✅ Modelo cargado: {selected_model_name}")

# Inputs del usuario
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

    
    # Construimos las features y realizamos la predicción
    X_user = build_full_features(inputData, model_columns=model_columns)
    prob = model.predict_proba(X_user)[0][1]
    st.success(f"Probabilidad de supervivencia: {prob:.2%}")

    # Intervalos de confianza
    try:
        lower, upper = bootstrap_ci(model, X_user)
        st.subheader("Intervalos de confianza")
        st.write(f"95% CI: [{lower[0]:.2%}, {upper[0]:.2%}]")
    except Exception as e:
        st.warning(f"No se pudo calcular el intervalo de confianza: {e}")


    # SHAP
    st.subheader("Explicación de la predicción (SHAP)")

    try:
        # Archivo de fondo (background) para SHAP
        pages_dir = os.path.dirname(__file__)
        project_dir = os.path.dirname(os.path.dirname(pages_dir))
        background_csv = os.path.join(project_dir, "data", "processed", "titanic_dataset_features.csv")

        if os.path.exists(background_csv):
            bg_df = pd.read_csv(background_csv)
            bg_df = bg_df.reindex(columns=model_columns, fill_value=0)
            bg_sample = bg_df.sample(n=min(100, len(bg_df)), random_state=42)  # Subsample para SHAP
        else:
            bg_sample = pd.DataFrame([np.zeros(len(model_columns))], columns=model_columns)

        X_user_1 = X_user.iloc[0:1]  # Solo 1 muestra para force plot

        # ---------------------
        # Diferente explainer dependiendo del tipo de modelo
        if isinstance(model, SVC):
            explainer = shap.KernelExplainer(model.predict_proba, bg_sample)
            shap_values_list = explainer.shap_values(X_user_1, nsamples=100)
            if isinstance(shap_values_list, list) and len(shap_values_list) > 1:
                shap_array = shap_values_list[1].reshape(-1)  # Clase positiva
                base_plot = explainer.expected_value[1]
            else:
                shap_array = np.array(shap_values_list).reshape(-1)
                base_plot = explainer.expected_value if isinstance(explainer.expected_value, (int, float)) else explainer.expected_value[0]

        elif isinstance(model, LogisticRegression):
            explainer = shap.LinearExplainer(model, bg_sample, feature_perturbation="interventional")
            shap_values = explainer.shap_values(X_user_1)
            shap_array = np.array(shap_values).reshape(-1)
            base_plot = explainer.expected_value

        else:  # Random Forest / XGBoost
            explainer = shap.Explainer(model, bg_sample)
            explanation = explainer(X_user_1)
            vals = explanation.values

            # Multi-clase
            if vals.ndim == 3:
                if vals.shape[1] > 1:
                    shap_array = vals[0, 1, :]
                    base_plot = explanation.base_values[0, 1]
                else:
                    shap_array = vals[0, 0, :]
                    base_plot = explanation.base_values[0, 0]
            elif vals.ndim == 2:
                shap_array = vals[0, :]
                base_plot = explanation.base_values[0] if hasattr(explanation.base_values, "__len__") else explanation.base_values
            else:
                shap_array = vals.flatten()
                base_plot = float(explanation.base_values)

        # ---------------------
        # Alineamos las dimensiones por si hay desalineación
        if shap_array.shape[0] != X_user_1.shape[1]:
            min_len = min(shap_array.shape[0], X_user_1.shape[1])
            shap_array = shap_array[:min_len]
            X_user_1 = X_user_1.iloc[:, :min_len]

        # ---------------------
        # Force plot de SHAP
        shap.initjs()
        fig_force = plt.figure(figsize=(12,4))
        shap.plots.force(base_plot, shap_array, X_user_1.iloc[0,:], matplotlib=True, show=False)
        st.pyplot(fig_force)
        plt.close(fig_force)

        # Summary plot (Importancia de features)
        fig_summary = plt.figure(figsize=(10,6))
        shap.summary_plot(shap_array.reshape(1,-1), X_user_1, plot_type="bar", show=False, max_display=10)
        st.pyplot(fig_summary)
        plt.close(fig_summary)

        # ---------------------
        # Tabla con top 10 features y contribución SHAP
        feature_importance = pd.DataFrame({
            "Característica": X_user_1.columns,
            "Valor": X_user_1.iloc[0].values,
            "Contribución SHAP": shap_array
        })
        feature_importance = feature_importance.reindex(
            feature_importance["Contribución SHAP"].abs().sort_values(ascending=False).index
        )
        st.dataframe(feature_importance.head(10))

    except Exception as e:
        st.warning(f"No se pudo generar SHAP: {e}")
        st.error(f"Detalles del error: {str(e)}")
