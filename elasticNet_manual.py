import csv
import random

#Función para cargar datos desde un archivo CSV
def cargar_datos(ruta_csv):
    """Carga datos desde un archivo CSV.
    Asume que la última columna es la etiqueta.
    Parametros:
        ruta_csv (str): Ruta al archivo CSV.
    Retorna:
        datos (list): Lista de filas, cada fila es una lista de valores (floats) y la última es la etiqueta (str)."""
    
    datos = []
    # Abrir el archivo CSV y leer los datos
    with open(ruta_csv, 'r') as archivo:
        lector = csv.reader(archivo)
        next(lector)  # Saltar encabezado
        for fila in lector:
            datos.append([float(x) for x in fila])
    
    return datos

# Función para separar características y etiquetas
def separar_X_y(datos):
    """Separa las características (X) y las etiquetas (y) de los datos.
    Parametros:
        datos (list): Lista de filas, cada fila es una lista de valores (floats) y la última es la etiqueta (str).
    Retorna:
        X (list): Lista de listas con las características.
        y (list): Lista con las etiquetas."""
    
    # Separar características y etiquetas
    X = [fila[:-1] for fila in datos]
    y = [fila[-1] for fila in datos]

    # Retornar tupla (X, y)
    return X, y

# Función para hacer predicciones lineales
def prediccion_lineal(X, w, b):
    """Hace predicciones lineales.
    Parametros:
        X (list): Lista de listas con las características.
        w (list): Lista de pesos.
        b (float): Sesgo.
    Retorna:
        list: Lista de predicciones."""
    # Seguimos la fórmula y = Xw + b
    return [sum(wi * xi for wi, xi in zip(w, x)) + b for x in X]

# Función para calcular el error cuadrático medio
def mse(y_true, y_pred):
    """Calcula el error cuadrático medio.
    Parametros:
        y_true (list): Lista de valores verdaderos.
        y_pred (list): Lista de valores predichos.
    Retorna:
        float: Error cuadrático medio."""
    
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)

# Función para calcular el error absoluto medio
def mae(y_true, y_pred):
    """Calcula el error absoluto medio (MAE)."""
    return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / len(y_true)

# Función para calcular el coeficiente de determinación R^2
def r2_score(y_true, y_pred):
    """Calcula el coeficiente de determinación R^2."""
    media = sum(y_true) / len(y_true)
    ss_tot = sum((yt - media) ** 2 for yt in y_true)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))

    return 1 - ss_res / ss_tot if ss_tot != 0 else 0

# Función para entrenar un modelo de regresión lineal con Elastic Net
def elastic_net_regression(X, y, alpha=0.1, l1_ratio=0.5, lr=0.01, epochs=1000):
    """Entrena un modelo de regresión lineal con regularización Elastic Net.
    Parametros:
        X (list): Lista de listas con las características.
        y (list): Lista de valores verdaderos.
        alpha (float): Tasa de regularización.
        l1_ratio (float): Proporción de L1 en Elastic Net (0 <= l1_ratio <= 1).
        lr (float): Tasa de aprendizaje.
        epochs (int): Número de épocas.
    Retorna:
        w (list): Lista de pesos entrenados.
        b (float): Sesgo entrenado.
    """
    # Inicialización de pesos y sesgo
    n, m = len(X), len(X[0])
    w = [random.random() for _ in range(m)]
    b = 0

    for _ in range(epochs):
        y_pred = prediccion_lineal(X, w, b)
        dw = []
        for j in range(m):
            l1 = l1_ratio * (1 if w[j] > 0 else -1)
            l2 = (1 - l1_ratio) * 2 * w[j]
            grad = (-2/n) * sum((yt - yp) * xi[j] for xi, yt, yp in zip(X, y, y_pred)) + alpha * (l1 + l2)
            dw.append(grad)
        db = (-2/n) * sum(yt - yp for yt, yp in zip(y, y_pred))
        w = [wi - lr*dwi for wi, dwi in zip(w, dw)]
        b -= lr * db
    return w, b

# Ejecución del código
if __name__ == "__main__":
    datos = cargar_datos('datos_regresion.csv')
    X, y = separar_X_y(datos)
    w_en, b_en = elastic_net_regression(X, y, alpha=0.1, l1_ratio=0.5, lr=0.01, epochs=1000)
    y_pred_en = prediccion_lineal(X, w_en, b_en)
    print("Elastic Net MSE:", mse(y, y_pred_en))
    print("Pesos:", w_en)
    print("Sesgo:", b_en)