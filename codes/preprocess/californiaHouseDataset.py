import csv
from sklearn.datasets import fetch_california_housing

# Cargar el dataset de California Housing
california = fetch_california_housing()
X = california.data  # Características
y = california.target  # Variable objetivo

# Guardar en un archivo CSV (sin encabezados)
with open('dataset/datos_california.csv', 'w', newline='') as archivo:
    escritor = csv.writer(archivo)
    for xi, yi in zip(X, y):
        fila = list(xi) + [yi]
        escritor.writerow(fila)

print("Archivo 'datos_california.csv' generado correctamente.")