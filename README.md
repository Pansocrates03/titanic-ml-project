# Titanic Machine Learning Project

Este proyecto tiene como objetivo desarrollar un sistema completo de análisis y predicción de supervivencia en el desastre del Titanic, utilizando técnicas de Machine Learning. El alcance del proyecto incluye la exploración de datos, preprocesamiento, ingeniería de características, entrenamiento y evaluación de modelos, análisis de sesgos, interpretabilidad de modelos y la creación de un dashboard interactivo para visualizar los resultados y permitir análisis "What-If".

## Estructura del Proyecto

El proyecto está organizado en las siguientes carpetas:

-   `data/`: Contiene los conjuntos de datos brutos y procesados utilizados en el proyecto.
    -   `raw/`: Datos originales del Titanic.
    -   `processed/`: Datos después del preprocesamiento y la ingeniería de características.
-   `notebooks/`: Cuadernos Jupyter para la exploración de datos, preprocesamiento, ingeniería de características, análisis de sesgos e interpretabilidad.
-   `models/`: Modelos de Machine Learning entrenados y serializados (e.g., `logistic_regression_final.pkl`).
-   `src/`: Contiene el código fuente de los módulos principales del proyecto (e.g., `preprocessing.py`, `features.py`, `akinator.py`).
-   `dashboard/`: Archivos relacionados con la aplicación web interactiva (dashboard) desarrollada con Streamlit.
-   `docs/`: Documentación del proyecto, incluyendo entregas y reportes.
-   `requirements.txt`: Lista de dependencias del proyecto.
-   `Dockerfile`: Archivo para la contenerización de la aplicación.

