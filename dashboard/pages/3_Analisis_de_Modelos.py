import streamlit as st
import pandas as pd
import json

# Este es un ejemplo de datos para las métricas y debe ser reemplazado por tus datos reales.
# Lo incluimos para que la aplicación sea ejecutable.
metrics = {
    "Logistic Regression": {
        "roc_auc": 0.85,
        "confusion_matrix": [
            [98, 12],
            [17, 52]
        ]
    },
    "SVM": {
        "roc_auc": 0.82,
        "confusion_matrix": [
            [95, 15],
            [20, 55]
        ]
    },
    "Random Forest": {
        "roc_auc": 0.89,
        "confusion_matrix": [
            [105, 5],
            [10, 65]
        ]
    },
    "XGBoost": {
        "roc_auc": 0.91,
        "confusion_matrix": [
            [108, 2],
            [8, 67]
        ]
    }
}

# Aquí se define un modelo predeterminado para el ejemplo.
selected_model_name = "XGBoost" 

# -----------------------------
# 1. Streamlit App
# -----------------------------
st.title("Sección de Análisis de Modelos")

st.subheader("Comparación de métricas entre modelos")

for metric in metrics:
    st.write(f"##$## Métricas para {metric}")
    st.write(f"**ROC-AUC:** {metrics[metric]['roc_auc']:.4f}")
    cm = metrics[metric]['confusion_matrix']
    cm_df = pd.DataFrame(cm, columns=["Pred: No Superviviente", "Pred: Superviviente"], index=["Real: No Superviviente", "Real: Superviviente"])
    st.dataframe(cm_df)

st.subheader("Visualización de feature importance")
st.subheader("Análisis de errores interactivo")
