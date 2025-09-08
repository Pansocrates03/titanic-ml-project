# Usa una imagen base de Python
FROM python:3.9-slim

# Establece el directorio de trabajo en /app
WORKDIR /app

# Copia el archivo requirements.txt al directorio de trabajo
COPY requirements.txt .

# Instala las dependencias del archivo requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de los archivos del proyecto al directorio de trabajo
COPY . .

# Expone el puerto por defecto de Streamlit
EXPOSE 8501

# Define el comando para ejecutar tu aplicación de Streamlit
# Asegúrate de que "dashboard/app.py" sea la ruta correcta a tu archivo principal
CMD ["streamlit", "run", "dashboard/app.py"]