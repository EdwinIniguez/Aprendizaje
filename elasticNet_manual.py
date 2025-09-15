import csv
import random
import math

# Función para cargar datos desde un archivo CSV
def cargar_datos(ruta_csv):
    """
    Carga datos desde un archivo CSV.
    Asume que la última columna es la etiqueta (valor a predecir).
    Parámetros:
        ruta_csv (str): Ruta al archivo CSV.
    Retorna:
        datos (list): Lista de filas, cada fila es una lista de valores (floats).
    """
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
    """
    Separa las características (X) y las etiquetas (y) de los datos.
    Parámetros:
        datos (list): Lista de filas, cada fila es una lista de valores (floats).
    Retorna:
        X (list): Lista de listas con las características.
        y (list): Lista con las etiquetas.
    """
    X = [fila[:-1] for fila in datos]  # Todas las columnas menos la última
    y = [fila[-1] for fila in datos]   # Última columna
    return X, y


# Función para hacer predicciones lineales
def prediccion_lineal(X, w, b):
    """
    Hace predicciones lineales usando los pesos y el sesgo.
    Fórmula: y = Xw + b
    Parámetros:
        X (list): Lista de listas con las características.
        w (list): Lista de pesos.
        b (float): Sesgo.
    Retorna:
        list: Lista de predicciones.
    """
    return [sum(wi * xi for wi, xi in zip(w, x)) + b for x in X]


# Función para calcular el error cuadrático medio (MSE)
def mse(y_true, y_pred):
    """
    Calcula el error cuadrático medio (MSE).
    Parámetros:
        y_true (list): Valores verdaderos.
        y_pred (list): Valores predichos.
    Retorna:
        float: Error cuadrático medio.
    """
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)


# Función para calcular el error absoluto medio (MAE)
def mae(y_true, y_pred):
    """
    Calcula el error absoluto medio (MAE).
    Parámetros:
        y_true (list): Valores verdaderos.
        y_pred (list): Valores predichos.
    Retorna:
        float: Error absoluto medio.
    """
    return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / len(y_true)


# Función para calcular el coeficiente de determinación R^2
def r2_score(y_true, y_pred):
    """
    Calcula el coeficiente de determinación R^2.
    Indica qué tan bien se ajusta el modelo a los datos.
    Parámetros:
        y_true (list): Valores verdaderos.
        y_pred (list): Valores predichos.
    Retorna:
        float: Valor de R^2.
    """
    media = sum(y_true) / len(y_true)
    ss_tot = sum((yt - media) ** 2 for yt in y_true)  # Suma total de cuadrados
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))  # Suma de residuos

    return 1 - ss_res / ss_tot if ss_tot != 0 else 0


# Función para entrenar un modelo de regresión lineal con Elastic Net
def elastic_net_regression(X, y, alpha=0.1, l1_ratio=0.5, lr=0.01, epochs=1000):
    """
    Entrena un modelo de regresión lineal con regularización Elastic Net usando descenso por gradiente.
    Parámetros:
        X (list): Características de entrada.
        y (list): Valores verdaderos.
        alpha (float): Tasa de regularización.
        l1_ratio (float): Proporción de L1 en Elastic Net (0 <= l1_ratio <= 1).
        lr (float): Tasa de aprendizaje.
        epochs (int): Número de épocas de entrenamiento.
    Retorna:
        w (list): Pesos entrenados.
        b (float): Sesgo entrenado.
    """
    n, m = len(X), len(X[0])  # n: número de muestras, m: número de características
    w = [random.random() for _ in range(m)]  # Inicializar pesos aleatoriamente
    b = 0  # Inicializar sesgo en 0

    for _ in range(epochs):
        y_pred = prediccion_lineal(X, w, b)  # Calcular predicciones actuales
        dw = []
        for j in range(m):
            # Gradiente para Elastic Net: combinación de L1 y L2
            l1 = l1_ratio * (1 if w[j] > 0 else -1)  # Derivada de L1
            l2 = (1 - l1_ratio) * 2 * w[j]           # Derivada de L2
            
            # Gradiente total para el peso j
            grad = (-2/n) * sum((yt - yp) * xi[j] for xi, yt, yp in zip(X, y, y_pred)) + alpha * (l1 + l2)
            dw.append(grad)
        
        # Gradiente para el sesgo
        db = (-2/n) * sum(yt - yp for yt, yp in zip(y, y_pred))

        # Actualizar pesos y sesgo
        w = [wi - lr*dwi for wi, dwi in zip(w, dw)]
        b -= lr * db
    return w, b


# Función para validación cruzada k-fold
def k_fold_cross_validation(X, y, k=5, **modelo_params):
    """
    Realiza validación cruzada k-fold para Elastic Net.
    Parámetros:
        X (list): Características.
        y (list): Etiquetas.
        k (int): Número de folds.
        modelo_params: Parámetros para elastic_net_regression.
    Retorna:
        dict: Métricas promedio (MSE, MAE, R2).
    """
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)
    fold_size = math.ceil(n / k)
    mse_scores, mae_scores, r2_scores = [], [], []

    for fold in range(k):
        # Crear índices para validación y entrenamiento
        val_indices = indices[fold*fold_size : (fold+1)*fold_size]
        train_indices = [i for i in indices if i not in val_indices]

        X_train = [X[i] for i in train_indices]
        y_train = [y[i] for i in train_indices]
        X_val = [X[i] for i in val_indices]
        y_val = [y[i] for i in val_indices]

        # Entrenar modelo
        w, b = elastic_net_regression(X_train, y_train, **modelo_params)
        y_pred = prediccion_lineal(X_val, w, b)

        # Calcular métricas
        mse_scores.append(mse(y_val, y_pred))
        mae_scores.append(mae(y_val, y_pred))
        r2_scores.append(r2_score(y_val, y_pred))

    # Promediar métricas
    return {
        "MSE_promedio": sum(mse_scores) / k,
        "MAE_promedio": sum(mae_scores) / k,
        "R2_promedio": sum(r2_scores) / k
    }


# Ejecución principal del código
if __name__ == "__main__":
    # Cargar los datos desde un archivo CSV
    datos = cargar_datos('dataset/datos_regresion_escalado.csv')

    # Separar características y etiquetas
    X, y = separar_X_y(datos)

    # Entrenar el modelo Elastic Net
    w_en, b_en = elastic_net_regression(X, y, alpha=0.1, l1_ratio=0.5, lr=0.01, epochs=1000)

    # Hacer predicciones con el modelo entrenado
    y_pred_en = prediccion_lineal(X, w_en, b_en)

    # Imprimir métricas de desempeño
    print("Elastic Net MSE:", mse(y, y_pred_en))
    print("Elastic Net MAE:", mae(y, y_pred_en))
    print("Elastic Net R2:", r2_score(y, y_pred_en))
    print("Pesos:", w_en)
    print("Sesgo:", b_en)

    # Validación cruzada k-fold
    resultados_cv = k_fold_cross_validation(X, y, k=5, alpha=0.1, l1_ratio=0.5, lr=0.01, epochs=1000)
    print("\nValidación cruzada (5-fold):")
    print("MSE promedio:", resultados_cv["MSE_promedio"])
    print("MAE promedio:", resultados_cv["MAE_promedio"])
    print("R2 promedio:", resultados_cv["R2_promedio"])