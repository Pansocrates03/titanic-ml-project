# Dashboard

Este dashboard interactivo ha sido desarrollado para explorar, predecir y analizar la supervivencia de los pasajeros del Titanic. Utiliza modelos de Machine Learning para ofrecer una visión completa y permitir simulaciones de escenarios (What-If).

## Correr 
1. Iniciar el dashboard
```shell
streamlit run Exploracion.py
```

Una vez que se ejecuta el comando, se abrirá una nueva pestaña en tu navegador web con la interfaz del dashboard.

## Páginas del Dashboard

El dashboard se compone de las siguientes secciones:

### Exploracion.py
Esta página permite una exploración interactiva de los datos del Titanic. Los usuarios pueden visualizar distribuciones de características, correlaciones y tendencias clave para entender mejor los factores que influyeron en la supervivencia.

### Prediccion.py
En esta sección, los usuarios pueden ingresar datos específicos de un pasajero para obtener una predicción de supervivencia utilizando los modelos de Machine Learning entrenados. Es una herramienta práctica para ver cómo los diferentes atributos impactan el resultado.

### Analisis_de_Modelos.py
Esta página ofrece un análisis detallado de los modelos de Machine Learning utilizados. Se presentan métricas de rendimiento, curvas ROC, matrices de confusión y otras visualizaciones para evaluar la calidad y la interpretabilidad de cada modelo.

### What-If.py
La sección "What-If" permite a los usuarios modificar los atributos de un pasajero y observar cómo estos cambios afectan la predicción de supervivencia. Es ideal para comprender la sensibilidad de los modelos a diferentes escenarios y realizar análisis contrafactuales.