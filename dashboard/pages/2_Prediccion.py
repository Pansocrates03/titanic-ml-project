import streamlit as st
import joblib
import numpy as np

def bootstrap_ci(model, X, n_bootstrap=1000, alpha=0.05):
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

@st.cache_resource
def load_model():
    return joblib.load("D:/GIT/titanic-ml-project/models/logistic_regression_final.pkl")

logistic_regression_model = load_model()

input_type = {
    "Age": float,
    "SibSp": int,
    "Parch": int,
    "Fare": float,
    "Sex_female": int,
    "Sex_male": int,
    "Embarked_C": int,
    "Embarked_Q": int,
    "Embarked_S": int,
    "Pclass_1": int,
    "Pclass_2": int,
    "Pclass_3": int
}

def predict_survival(input_data):
    final_fare = 32
    if input_data["clase"] == "Primera":
        final_fare = 52
    elif input_data["clase"] == "Segunda":
        final_fare = 12
    elif input_data["clase"] == "Tercera":
        final_fare = 8

    from typing import Dict, Any
    data: Dict[str, Any] = {
        "Age": input_data["age"],
        "SibSp": input_data["SibSp"],
        "Parch": input_data["parch"],
        "Fare": final_fare,
        "Sex_female": 1 if input_data["sex"] == "Femenino" else 0,
        "Sex_male": 1 if input_data["sex"] == "Masculino" else 0,
        "Embarked_C": 0,
        "Embarked_Q": 1,
        "Embarked_S": 0,
        "Pclass_1": 1 if input_data["clase"] == "Primera" else 0,
        "Pclass_2": 1 if input_data["clase"] == "Segunda" else 0,
        "Pclass_3": 1 if input_data["clase"] == "Tercera" else 0
    }
    features = np.array([list(data.values())])
    prediction = logistic_regression_model.predict_proba(features)[0][1]
    return prediction, data

st.title("🚢 Predicción de Supervivencia en el Titanic")
st.write("¿Sobrevivirías al Titanic? Ingresa tus datos:")

inputData = {
    "name": st.text_input("Nombre", "John Doe"),
    "age": st.number_input("Edad", min_value=0, max_value=100, value=30),
    "parch": st.number_input("Padres o hijos a bordo", min_value=0, max_value=10, value=0),
    "SibSp": st.number_input("Hermanos a bordo", min_value=0, max_value=10, value=0),
    "clase": st.selectbox("Clase", ["Primera", "Segunda", "Tercera"]),
    "sex": st.selectbox("Sexo", ["Masculino", "Femenino"])
}

if st.button("Predecir Supervivencia", key="predict_button"):
    prob, data = predict_survival(inputData)
    st.success(f"Probabilidad de supervivencia: {prob:.2%}")

    # Calcular intervalo de confianza usando bootstrap con los datos del usuario
    import pandas as pd
    X_user = pd.DataFrame([data])
    lower, upper = bootstrap_ci(logistic_regression_model, X_user)
    st.subheader("Intervalos de confianza")
    st.write(f"Intervalo de confianza del 95%: [{lower[0]:.2%}, {upper[0]:.2%}]")
else:
    st.subheader("Intervalos de confianza")
    st.write("Ingresar datos y hacer clic en 'Predecir Supervivencia' para ver los intervalos de confianza.")

# ========================
# 4. Explicaciones adicionales
# ========================
st.subheader("Explicación de la predicción (SHAP/LIME)")
st.info("La predicción que realiza tu aplicación depende de tres factores principales: el sexo, la edad y la clase en la que viajaba la persona. Estos factores influyen en la probabilidad de supervivencia según el modelo entrenado. La combinación de estos factores influye en la predicción porque, históricamente, ciertos grupos (como mujeres, niños o pasajeros de primera clase) tenían mayor probabilidad de sobrevivir.")
