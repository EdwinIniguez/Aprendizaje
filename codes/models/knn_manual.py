import csv
import math
import random

from collections import Counter

#Función para cargar datos desde un archivo CSV
def cargar_datos(ruta_csv):
    """Carga datos desde un archivo CSV.
    Asume que la última columna es la etiqueta.
    Parametros:
        ruta_csv (str): Ruta al archivo CSV.
    Retorna:
        datos (list): Lista de filas, cada fila es una lista de valores (floats) y la última es la etiqueta (str)."""
    
    datos = []
    with open(ruta_csv, 'r') as archivo:
        lector = csv.reader(archivo)
        next(lector)  # Saltar encabezado si existe
        
        for fila in lector:
            # Convierte todas las columnas menos la última a float, la última es la etiqueta
            datos.append([float(x) for x in fila[:-1]] + [fila[-1]])

    return datos

# Función para calcular la distancia euclidiana entre dos puntos
def distancia_euclidiana(p1, p2):
    """Calcula la distancia euclidiana entre dos puntos.
    Parametros:
        p1 (list): Primer punto.
        p2 (list): Segundo punto.
    Retorna:
        float: Distancia euclidiana."""
    
    # Seguimos la fórmula de distancia euclidiana
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2))) # Raíz cuadrada de la suma de las diferencias al cuadrado

def knn_clasificar(datos_entrenamiento, ejemplo, k=3):
    # Calcular distancia a todos los puntos de entrenamiento
    distancias = []
    
    for fila in datos_entrenamiento:
        x = fila[:-1]
        etiqueta = fila[-1]
        dist = distancia_euclidiana(x, ejemplo)
        distancias.append((dist, etiqueta))

    # Ordenar por distancia y tomar los k más cercanos
    vecinos = sorted(distancias, key=lambda x: x[0])[:k]
    
    # Contar la clase más frecuente
    etiquetas = [etiqueta for _, etiqueta in vecinos]
    prediccion = Counter(etiquetas).most_common(1)[0][0]
    return prediccion

# Funciones para calcular métricas de evaluación
def accuracy(y_true, y_pred):
    """Calcula la accuracy de las predicciones.
    Parametros:
        y_true (list): Etiquetas verdaderas.
        y_pred (list): Etiquetas predichas.
    Retorna:
        float: Precisión."""
    return sum(yt == yp for yt, yp in zip(y_true, y_pred)) / len(y_true)

def precision(y_true, y_pred, clase):
    """Calcula la precision de las predicciones para una clase específica.
    Parametros:
        y_true (list): Etiquetas verdaderas.
        y_pred (list): Etiquetas predichas.
        clase: Clase para la cual calcular la precisión.
    Retorna:
        float: Precisión."""
    tp = sum((yt == clase and yp == clase) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt != clase and yp == clase) for yt, yp in zip(y_true, y_pred))
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall(y_true, y_pred, clase):
    """Calcula el recall de las predicciones para una clase específica.
    Parametros:
        y_true (list): Etiquetas verdaderas.
        y_pred (list): Etiquetas predichas.
        clase: Clase para la cual calcular el recall.
    Retorna:
        float: Recall."""
    tp = sum((yt == clase and yp == clase) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == clase and yp != clase) for yt, yp in zip(y_true, y_pred))
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def f1_score(y_true, y_pred, clase):
    """Calcula el F1-score de las predicciones para una clase específica.
    Parametros:
        y_true (list): Etiquetas verdaderas.
        y_pred (list): Etiquetas predichas.
        clase: Clase para la cual calcular el F1-score.
    Retorna:
        float: F1-score."""
    p = precision(y_true, y_pred, clase)
    r = recall(y_true, y_pred, clase)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0

def reporte_metricas(y_true, y_pred):
    """Imprime un reporte de métricas de evaluación.
    Parametros:
        y_true (list): Etiquetas verdaderas.
        y_pred (list): Etiquetas predichas."""
    clases = set(y_true)
    print(f"Accuracy: {accuracy(y_true, y_pred):.2f}")
    for clase in clases:
        p = precision(y_true, y_pred, clase)
        r = recall(y_true, y_pred, clase)
        f1 = f1_score(y_true, y_pred, clase)
        print(f"Clase {clase}: Precision={p:.2f}, Recall={r:.2f}, F1={f1:.2f}")

# Ejecución principal
if __name__ == "__main__":
    # Cambia 'datos.csv' por tu archivo de datos
    datos = cargar_datos('dataset/datos_regresion.csv')

    # Divide en entrenamiento y prueba (aquí, los últimos 5 para prueba)
    datos_entrenamiento = datos[:-5]
    datos_prueba = datos[-5:]

    aciertos = 0
    for fila in datos_prueba:
        x = fila[:-1]
        etiqueta_real = fila[-1]
        prediccion = knn_clasificar(datos_entrenamiento, x, k=3)
        print(f"Real: {etiqueta_real} - Predicho: {prediccion}")
        if prediccion == etiqueta_real:
            aciertos += 1
    print(f"Accuracy: {aciertos/len(datos_prueba):.2f}")