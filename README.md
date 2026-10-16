# Clasificación de imágenes con una CNN (CIFAR-10)

Proyecto de Deep Learning que clasifica imágenes en diez categorías usando una red neuronal convolucional (CNN), TensorFlow/Keras y CIFAR-10.

## Dataset

CIFAR-10 contiene 60.000 imágenes RGB de 32 × 32 píxeles: 50.000 imágenes de entrenamiento original y 10.000 de prueba. El notebook separa una parte del conjunto original para validación y conserva el conjunto de prueba para la evaluación final.

Clases: avión, automóvil, pájaro, gato, ciervo, perro, rana, caballo, barco y camión.

## Estructura

- `notebooks/experimentos_cnn.ipynb`: exploración, visualización, preprocesamiento, división, Data Augmentation, CNN, entrenamiento, gráficas, métricas, matriz de confusión y 10 predicciones visuales.
- `src/entrenar.py`: script para entrenar el modelo desde un entorno Python.
- `src/predecir.py`: función reutilizable para predecir una imagen.
- `modelo/`: aquí deben quedar `cnn_cifar10.keras` y `class_names.json`, generados por el script de entrenamiento.
- `app.py`: aplicación Streamlit.
- `requirements.txt`: dependencias.

## 1. Ejecutar en Google Colab

1. Abre Google Colab y sube `notebooks/experimentos_cnn.ipynb`.
2. Opcionalmente, activa una GPU desde Entorno de ejecución > Cambiar tipo de entorno de ejecución.
3. Ejecuta las celdas en orden, hasta el final.
4. Se pueden ver las gráficas, el reporte de clasificación, la matriz de confusión y las diez predicciones.
5. La última celda descarga `modelo_cnn_cifar10.zip`.


## 2. Ejecutar Streamlit

Con los archivos:

- `modelo/cnn_cifar10.keras`
- `modelo/class_names.json`

Luego, desde la carpeta raíz del proyecto: (Recomiendo por experiencia propia en un entorno vritual python 3.12)

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit mostrará una dirección local en la terminal. Ábrela en el navegador, carga una imagen JPG, JPEG, PNG o WEBP y pulsa “Realizar predicción”.


## Consideraciones

- El modelo se entrena con CIFAR-10, imágenes pequeñas de 32 × 32 y diez clases.
- Streamlit redimensiona la imagen cargada y normaliza sus píxeles igual que durante el entrenamiento.
- Una imagen que no corresponda a las clases del dataset puede recibir una predicción incorrecta con confianza alta.
