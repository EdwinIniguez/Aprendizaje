import numpy as np
import csv
import joblib
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
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
    datos = cargar_datos('dataset/datos_california.csv')
    X, y = separar_X_y(datos)

    # Pipeline con escalado y ElasticNet
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('elasticnet', ElasticNet(max_iter=10000))
    ])

    # Definir el grid de hiperparámetros
    param_grid = {
        'elasticnet__alpha': [0.001, 0.01, 0.1, 1, 10],
        'elasticnet__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    }

    # GridSearchCV con validación cruzada
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring='r2',
        n_jobs=-1
    )
    grid.fit(X, y)

    print("Mejores hiperparámetros encontrados:")
    print(grid.best_params_)
    print("\nMejor R2 promedio (cross-validation):", grid.best_score_)

    # Evaluar el mejor modelo en todo el dataset
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X)
    print("\nResultados del mejor modelo en todo el dataset:")
    print("  MSE:", mean_squared_error(y, y_pred))
    print("  MAE:", mean_absolute_error(y, y_pred))
    print("  R2 :", r2_score(y, y_pred))
    print("  Pesos:", best_model.named_steps['elasticnet'].coef_)
    print("  Sesgo:", best_model.named_steps['elasticnet'].intercept_)

    # Guardar el modelo
    joblib.dump(best_model, 'models/elasticnet_california.joblib')