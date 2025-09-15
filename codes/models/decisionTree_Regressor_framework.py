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

def train_modelo(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def evaluar_modelo(model, X, y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return mse, mae, r2

def validacion_cruzada(model, X, y, k=5):
    cv = KFold(n_splits=k, shuffle=True, random_state=42)
    resultados = cross_validate(
        model, X, y, cv=cv,
        scoring=('neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'),
        return_train_score=False
    )
    return resultados

def imprimir_resultados(resultados):
    print("Resultados de validación cruzada:")
    print("  MSE promedio:", -np.mean(resultados['test_neg_mean_squared_error']))
    print("  MAE promedio:", -np.mean(resultados['test_neg_mean_absolute_error']))
    print("  R2  promedio:", np.mean(resultados['test_r2']))

def guardar_modelo(model, ruta):
    import joblib
    joblib.dump(model, ruta)


def main():
    datos = cargar_datos('dataset/datos_california.csv')
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
    modelo = train_modelo(pipeline, X_train, y_train)

    # Evaluar en entrenamiento
    mse, mae, r2 = evaluar_modelo(modelo, X_train, y_train)
    print("\nResultados en entrenamiento:")
    print("  MSE:", mse)
    print("  MAE:", mae)
    print("  R2 :", r2)
    
    # Evaluar en prueba
    mse, mae, r2 = evaluar_modelo(modelo, X_test, y_test)
    print("\nResultados en prueba:")
    print("  MSE:", mse)
    print("  MAE:", mae)
    print("  R2 :", r2)

    # Validación cruzada k-fold
    k = 5
    resultados = validacion_cruzada(modelo, X, y, k)
    print(f"\nValidación cruzada ({k}-fold):")
    print("  MSE promedio:", -np.mean(resultados['test_neg_mean_squared_error']))
    print("  MAE promedio:", -np.mean(resultados['test_neg_mean_absolute_error']))
    print("  R2  promedio:", np.mean(resultados['test_r2']))

    # Guardar el modelo entrenado
    guardar_modelo(modelo, 'models/modelo_decision_tree.joblib')

if __name__ == "__main__":
    main()