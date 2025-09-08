from matplotlib import pyplot as plt
import streamlit as st
import joblib
import numpy as np
import os
import pandas as pd
import shap
import pkg_resources
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


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

    # Reindexar según columnas del modelo para evitar errores
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
        # Elige el explainer y calcula los valores SHAP
        if isinstance(model, (xgb.XGBClassifier, RandomForestClassifier)):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_user)
            
            # Para modelos de árbol, shap_values puede ser una lista o array
            if isinstance(shap_values, list):
                # Para clasificación binaria, tomar la clase positiva (índice 1)
                positive_class_shap_values = shap_values[1]
                base_value = explainer.expected_value[1]
            else:
                # Si es un array 2D, tomar la segunda columna (clase positiva)
                if len(shap_values.shape) > 1 and shap_values.shape[1] > 1:
                    positive_class_shap_values = shap_values[:, 1]
                    base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
                else:
                    positive_class_shap_values = shap_values
                    base_value = explainer.expected_value
                    
        elif isinstance(model, LogisticRegression):
            explainer = shap.LinearExplainer(model, X_user, feature_perturbation="interventional")
            shap_values = explainer.shap_values(X_user)
            positive_class_shap_values = shap_values
            base_value = explainer.expected_value
            
        elif isinstance(model, SVC):
            # Usar KernelExplainer para SVM
            explainer = shap.KernelExplainer(model.predict_proba, X_user)
            shap_values = explainer.shap_values(X_user)
            
            if isinstance(shap_values, list):
                positive_class_shap_values = shap_values[1]
                base_value = explainer.expected_value[1]
            else:
                positive_class_shap_values = shap_values[:, 1]
                base_value = explainer.expected_value[1]
        else:
            raise ValueError("Tipo de modelo no soportado para SHAP.")

        shap.initjs()

        # 1. Generar el force_plot con la sintaxis corregida
        fig_force = plt.figure(figsize=(12, 4))
        
        # Asegurar que tenemos los valores correctos
        if isinstance(positive_class_shap_values, np.ndarray) and len(positive_class_shap_values.shape) > 1:
            shap_vals_for_plot = positive_class_shap_values[0, :]
        else:
            shap_vals_for_plot = positive_class_shap_values[0] if len(positive_class_shap_values) > 0 else positive_class_shap_values
        
        # Sintaxis corregida para SHAP v0.20+
        shap.plots.force(
            base_value,
            shap_vals_for_plot,
            X_user.iloc[0, :],
            matplotlib=True,
            show=False
        )
        st.pyplot(fig_force, bbox_inches='tight', dpi=150, pad_inches=0.1)
        plt.close(fig_force)
        
        # 2. Generar el summary_plot (bar plot)
        fig_summary = plt.figure(figsize=(10, 6))
        shap.summary_plot(
            positive_class_shap_values,
            X_user,
            plot_type="bar",
            show=False,
            max_display=10  # Mostrar solo las 10 características más importantes
        )
        st.pyplot(fig_summary)
        plt.close(fig_summary)

        # 3. Tabla con valores SHAP para referencia
        st.subheader("Contribuciones de las características")
        if isinstance(positive_class_shap_values, np.ndarray) and len(positive_class_shap_values.shape) > 1:
            shap_vals = positive_class_shap_values[0, :]
        else:
            shap_vals = positive_class_shap_values[0] if len(positive_class_shap_values) > 0 else positive_class_shap_values
            
        feature_importance = pd.DataFrame({
            'Característica': X_user.columns,
            'Valor': X_user.iloc[0].values,
            'Contribución SHAP': shap_vals
        })
        feature_importance = feature_importance.reindex(
            feature_importance['Contribución SHAP'].abs().sort_values(ascending=False).index
        )
        st.dataframe(feature_importance.head(10))

    except Exception as e:
        st.warning(f"No se pudo generar SHAP: {e}")
        st.error(f"Detalles del error: {str(e)}")

else:
    st.subheader("Intervalos de confianza")
    st.write("👉 Ingresa datos y haz clic en 'Predecir Supervivencia'.")