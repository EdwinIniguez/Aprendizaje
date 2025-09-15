import numpy as np
import csv
from sklearn.linear_model import ElasticNet
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_validate, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def cargar_datos(ruta_csv):
    """Carga datos desde un archivo CSV."""
    datos = []
    with open(ruta_csv, 'r') as archivo:
        lector = csv.reader(archivo)
        for fila in lector:
            datos.append([float(x) for x in fila])
    return datos

def separar_X_y(datos):
    """Separa características y etiquetas."""
    X = [fila[:-1] for fila in datos]
    y = [fila[-1] for fila in datos]
    return np.array(X), np.array(y)

def crear_pipeline(alpha=0.1, l1_ratio=0.5):
    """Crea una pipeline con escalado y ElasticNet."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('elasticnet', ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=1000))
    ])

if __name__ == "__main__":
    datos = cargar_datos('dataset/datos_regresion.csv')
    X, y = separar_X_y(datos)

    # Crear pipeline
    pipeline = crear_pipeline(alpha=0.1, l1_ratio=0.5)
    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)

    print("Resultados en todo el dataset:")
    print("  MSE:", mean_squared_error(y, y_pred))
    print("  MAE:", mean_absolute_error(y, y_pred))
    print("  R2 :", r2_score(y, y_pred))
    print("  Pesos:", pipeline.named_steps['elasticnet'].coef_)
    print("  Sesgo:", pipeline.named_steps['elasticnet'].intercept_)

    # Validación cruzada k-fold usando cross_validate
    k = 5
    cv = KFold(n_splits=k, shuffle=True, random_state=42)
    scoring = {
        'mse': make_scorer(mean_squared_error),
        'mae': make_scorer(mean_absolute_error),
        'r2': make_scorer(r2_score)
    }
    resultados = cross_validate(
        pipeline, X, y, cv=cv, scoring=scoring, return_train_score=False
    )

    print(f"\nValidación cruzada ({k}-fold):")
    print("  MSE promedio:", np.mean(resultados['test_mse']))
    print("  MAE promedio:", np.mean(resultados['test_mae']))
    print("  R2  promedio:", np.mean(resultados['test_r2']))