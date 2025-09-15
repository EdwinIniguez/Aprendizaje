import numpy as np
import csv
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.pipeline import Pipeline

def cargar_datos(ruta_csv):
    datos = []
    with open(ruta_csv, 'r') as archivo:
        lector = csv.reader(archivo)
        for fila in lector:
            datos.append([float(x) for x in fila])
    return datos

def separar_X_y(datos):
    X = [fila[:-1] for fila in datos]
    y = [fila[-1] for fila in datos]
    return np.array(X), np.array(y)

if __name__ == "__main__":
    datos = cargar_datos('dataset/datos_regresion.csv')
    X, y = separar_X_y(datos)

    # Separar en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Crear pipeline con DecisionTreeRegressor
    pipeline = Pipeline([
        ('tree', DecisionTreeRegressor(random_state=42))
    ])

    # Entrenar el modelo
    pipeline.fit(X_train, y_train)
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    print("\nResultados en prueba:")
    print("  MSE:", mean_squared_error(y_test, y_pred_test))
    print("  MAE:", mean_absolute_error(y_test, y_pred_test))
    print("  R2 :", r2_score(y_test, y_pred_test))

    # Validación cruzada k-fold
    k = 5
    cv = KFold(n_splits=k, shuffle=True, random_state=42)
    resultados = cross_validate(
        pipeline, X, y, cv=cv,
        scoring=('neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'),
        return_train_score=False
    )

    print(f"\nValidación cruzada ({k}-fold):")
    print("  MSE promedio:", -np.mean(resultados['test_neg_mean_squared_error']))
    print("  MAE promedio:", -np.mean(resultados['test_neg_mean_absolute_error']))
    print("  R2  promedio:", np.mean(resultados['test_r2']))