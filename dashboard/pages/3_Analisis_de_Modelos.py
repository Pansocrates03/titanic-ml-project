import streamlit as st
import pandas as pd
import json
import numpy as np
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


# -----------------------------
# 0. Configuración inicial
# -----------------------------

model, model_columns = load_model("logistic_regression_final.pkl")

# Este es un ejemplo de datos para las métricas y debe ser reemplazado por tus datos reales.
# Lo incluimos para que la aplicación sea ejecutable.
metrics = {
    "Logistic Regression": {
        "roc_auc": 0.85,
        "confusion_matrix": [
            [98, 12],
            [17, 52]
        ],
        "pre_rec_f1_sup": [
            [0.85, 0.89, 0.87, 110],
            [0.81, 0.75, 0.78, 69]
        ]
    },
    "SVM": {
        "roc_auc": 0.82,
        "confusion_matrix": [
            [98, 12],
            [18, 51]
        ],
        "pre_rec_f1_sup": [
            [0.84, 0.89, 0.87, 110],
            [0.81, 0.74, 0.77, 69]
        ]
    },
    "Random Forest": {
        "roc_auc": 0.89,
        "confusion_matrix": [
            [95, 15],
            [22, 47]
        ],
        "pre_rec_f1_sup": [
            [0.79, 0.77, 0.78, 179],
            [0.79, 0.79, 0.79, 179]
        ]
    },
    "XGBoost": {
        "roc_auc": 0.91,
        "confusion_matrix": [
            [98, 12],
            [18, 51]
        ],
        "pre_rec_f1_sup": [
            [0.82, 0.89, 0.85, 110],
            [0.81, 0.74, 0.77, 69]
        ]
    }
}

feature_importance = {
    "Title_Master": 1.369254,
    "Title_Mr": 1.251902,
    "Sex_female": 1.148413,
    "CabinDeck_E": 0.825982,
    "CabinDeck_D": 0.810746,
    "CabinDeck_Unknown": 0.806829,
    "Pclass_3": 0.737909,
    "Title_Mrs": 0.517539,
    "FamilySize": 0.510012,
    "Sex_male": 0.431202
}

# Aquí se define un modelo predeterminado para el ejemplo.
selected_model_name = "XGBoost" 

# --- Simulación de modelos y datos de prueba ---
# Usamos una clase simple para simular un modelo y así poder usar SHAP
class MockModel:
    def __init__(self):
        # El modelo no hace nada, solo devuelve probabilidades simuladas
        pass
    def predict(self, X):
        return np.random.randint(0, 2, size=len(X))
    def predict_proba(self, X):
        return np.random.rand(len(X), 2)

model = MockModel()

# Características del modelo de ejemplo
model_columns = ["Age","SibSp","Parch","FamilySize","IsAlone","FarePerPerson",
"Sex_female","Sex_male", "Pclass_1","Pclass_2","Pclass_3",
"Title_Mr","Title_Mrs","Title_Miss","Title_Master","Title_Ms",
"Title_Dr","Title_Rev","Title_Col","Title_Major","Title_Capt",
"Title_Sir","Title_Lady","Title_Jonkheer","Title_Dona","Title_Countess",
"AgeGroup_Child","AgeGroup_Teen","AgeGroup_Adult","AgeGroup_Senior"]

# Simular datos de prueba
X_test = pd.DataFrame(np.random.rand(200, len(model_columns)), columns=model_columns)
y_test = pd.Series(np.random.randint(0, 2, 200))
y_pred = pd.Series(model.predict(X_test))


# -----------------------------
# 1. Streamlit App
# -----------------------------
st.title("Sección de Análisis de Modelos")

st.subheader("Comparación de métricas entre modelos")

for metric in metrics:
    st.write(f"#### Métricas para {metric}")
    st.write(f"**ROC-AUC:** {metrics[metric]['roc_auc']:.4f}")

    # Mostrar la matriz de confusión
    cm = metrics[metric]['confusion_matrix']
    cm_df = pd.DataFrame(cm, columns=["Pred: No Superviviente", "Pred: Superviviente"], index=["Real: No Superviviente", "Real: Superviviente"])
    st.dataframe(cm_df)

    # Mostrar precisión, recall, f1-score y soporte
    prf = metrics[metric]['pre_rec_f1_sup']
    prf_df = pd.DataFrame(prf, columns=["Precisión", "Recall", "F1-Score", "Soporte"], index=["Clase 0", "Clase 1"])
    st.dataframe(prf_df)

st.subheader("Visualización de feature importance")
st.write(f"Mostrando feature importance para un modelo de ejemplo")
feature_importance_df = pd.DataFrame(list(feature_importance.items()), columns=["Característica", "Importancia"])
st.dataframe(feature_importance_df)

st.subheader("Análisis de errores interactivo")

if st.button("Analizar Errores"):
    st.write("### 🧠 Explorando errores del modelo...")

    # Identificar falsos positivos y falsos negativos
    false_negatives = X_test[(y_test == 1) & (y_pred == 0)]
    false_positives = X_test[(y_test == 0) & (y_pred == 1)]

    # 1. Falsos Negativos
    st.write("#### 🚩 Casos de Falsos Negativos (Pasajeros que sobrevivieron pero el modelo predijo que no)")
    if not false_negatives.empty:
        for i, row in false_negatives.iterrows():
            with st.expander(f"Pasajero {i}"):
                st.dataframe(row.to_frame().T)
                if st.button(f"Explicar predicción para pasajero {i}", key=f"fn_{i}"):
                    try:
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(row.to_frame().T)
                        shap.initjs()
                        st.write("#### Explicación con SHAP")
                        shap.force_plot(explainer.expected_value, shap_values[0], row.to_frame().T.iloc[0,:], matplotlib=True, show=False)
                        st.pyplot(bbox_inches='tight', dpi=150, pad_inches=0.1)
                    except Exception as e:
                        st.error(f"Error al generar SHAP: {e}")
    else:
        st.info("¡Excelente! No hay falsos negativos para este modelo en el conjunto de prueba.")

    # 2. Falsos Positivos
    st.write("#### 🔴 Casos de Falsos Positivos (Pasajeros que no sobrevivieron pero el modelo predijo que sí)")
    if not false_positives.empty:
        for i, row in false_positives.iterrows():
            with st.expander(f"Pasajero {i}"):
                st.dataframe(row.to_frame().T)
                if st.button(f"Explicar predicción para pasajero {i}", key=f"fp_{i}"):
                    try:
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(row.to_frame().T)
                        shap.initjs()
                        st.write("#### Explicación con SHAP")
                        shap.force_plot(explainer.expected_value, shap_values[0], row.to_frame().T.iloc[0,:], matplotlib=True, show=False)
                        st.pyplot(bbox_inches='tight', dpi=150, pad_inches=0.1)
                    except Exception as e:
                        st.error(f"Error al generar SHAP: {e}")
    else:
        st.info("¡Excelente! No hay falsos positivos para este modelo en el conjunto de prueba.")
