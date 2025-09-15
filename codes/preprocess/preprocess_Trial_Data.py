import csv
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

# Cargar el dataset de diabetes
diabetes = load_diabetes()
X = diabetes.data  # Características
y = diabetes.target  # Variable objetivo

# Guardar en un archivo CSV (sin encabezados)
with open('dataset/datos_regresion.csv', 'w', newline='') as archivo:
    escritor = csv.writer(archivo)
    for xi, yi in zip(X, y):
        fila = list(xi) + [yi]
        escritor.writerow(fila)

print("Archivo 'datos_regresion.csv' escalado generado correctamente.")