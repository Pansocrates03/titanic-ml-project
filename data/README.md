# Carpeta de Datos

Esta carpeta contiene todos los conjuntos de datos utilizados en el proyecto, organizados en subdirectorios para distinguir entre datos brutos (raw) y datos procesados. El objetivo es mantener una clara separación y trazabilidad de las transformaciones realizadas en los datos.

## Subdirectorio `raw`

Contiene los datos originales y sin modificar del conjunto de datos del Titanic.

- `Titanic-Dataset.csv`: El conjunto de datos original del Titanic, tal como fue descargado, antes de cualquier preprocesamiento o ingeniería de características.

## Subdirectorio `processed`

Contiene los conjuntos de datos que han sido preprocesados y a los que se les ha aplicado ingeniería de características, listos para el entrenamiento y evaluación de modelos.

- `titanic_dataset_features.csv`: Conjunto de datos con las características seleccionadas y transformadas, utilizado para el entrenamiento de los modelos.
- `Titanic-Dataset_with_features.csv`: Una versión del conjunto de datos del Titanic que incluye las nuevas características generadas durante la fase de ingeniería de características, junto con las columnas originales relevantes.

