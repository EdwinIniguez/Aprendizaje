import numpy as np
import csv
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

    # Pipeline con escalado y KNN Regressor
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsRegressor())
    ])

    # Definir el grid de hiperparámetros
    param_grid = {
        'knn__n_neighbors': [3, 5, 7, 9],
        'knn__weights': ['uniform', 'distance'],
        'knn__p': [1, 2]  # 1: Manhattan, 2: Euclidean
    }

    # GridSearchCV para encontrar la mejor configuración
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring='r2',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    print("Mejores hiperparámetros encontrados:")
    print(grid.best_params_)
    print("Mejor R2 promedio (cross-validation):", grid.best_score_)

    # Evaluar el mejor modelo en entrenamiento y prueba
    best_model = grid.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    print("\nResultados en entrenamiento:")
    print("  MSE:", mean_squared_error(y_train, y_pred_train))
    print("  MAE:", mean_absolute_error(y_train, y_pred_train))
    print("  R2 :", r2_score(y_train, y_pred_train))

    print("\nResultados en prueba:")
    print("  MSE:", mean_squared_error(y_test, y_pred_test))
    print("  MAE:", mean_absolute_error(y_test, y_pred_test))
    print("  R2 :", r2_score(y_test, y_pred_test))