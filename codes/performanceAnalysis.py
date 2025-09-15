import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, learning_curve, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import csv

# Cargar datos
def cargar_datos(ruta_csv):
    datos = []
    with open(ruta_csv, 'r') as archivo:
        for fila in csv.reader(archivo):
            datos.append([float(x) for x in fila])
    X = np.array([fila[:-1] for fila in datos])
    y = np.array([fila[-1] for fila in datos])
    return X, y

# Parámetros del modelo
alpha = 0.1
l1_ratio = 0.5

# Cargar y separar datos
X, y = cargar_datos('dataset/datos_regresion.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
modelo = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=1000, random_state=42)
modelo.fit(X_train, y_train)

# Predicciones
y_pred_train = modelo.predict(X_train)
y_pred_test = modelo.predict(X_test)

# Métricas
mse_train = mean_squared_error(y_train, y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

mse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print("Entrenamiento:")
print(f"  MSE: {mse_train:.2f} | MAE: {mae_train:.2f} | R2: {r2_train:.2f}")
print("Prueba:")
print(f"  MSE: {mse_test:.2f} | MAE: {mae_test:.2f} | R2: {r2_test:.2f}")

# =========================
# Gráficas para el reporte
# =========================

# 1. Curva de aprendizaje (train/validation)
train_sizes, train_scores, test_scores = learning_curve(
    modelo, X_train, y_train, cv=5, scoring='neg_mean_squared_error',
    train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
)
train_scores_mean = -np.mean(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)

plt.figure(figsize=(8,5))
plt.plot(train_sizes, train_scores_mean, 'o-', label="Entrenamiento")
plt.plot(train_sizes, test_scores_mean, 'o-', label="Validación")
plt.xlabel("Tamaño del conjunto de entrenamiento")
plt.ylabel("MSE")
plt.title("Curva de aprendizaje (Elastic Net)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("learning_curve.png")
plt.show()

# 2. Gráfico de predicciones vs valores reales (diagnóstico de bias/varianza)
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred_test, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Valor real")
plt.ylabel("Predicción")
plt.title("Predicción vs Real (conjunto de prueba)")
plt.grid()
plt.tight_layout()
plt.savefig("pred_vs_real.png")
plt.show()

# 3. Histograma de errores (residuos)
errores = y_test - y_pred_test
plt.figure(figsize=(8,5))
plt.hist(errores, bins=20, color='skyblue', edgecolor='black')
plt.xlabel("Error (residuo)")
plt.ylabel("Frecuencia")
plt.title("Distribución de errores (residuos) en prueba")
plt.grid()
plt.tight_layout()
plt.savefig("hist_errores.png")
plt.show()

# 4. Boxplot de métricas (train vs test)
plt.figure(figsize=(8,5))
plt.boxplot([[mse_train, mse_test], [mae_train, mae_test], [r2_train, r2_test]],
            labels=["MSE", "MAE", "R2"])
plt.title("Comparación de métricas (Train vs Test)")
plt.ylabel("Valor")
plt.tight_layout()
plt.savefig("boxplot_metricas.png")
plt.show()

# 5. Diagnóstico de underfit/overfit
print("\nDiagnóstico de ajuste del modelo:")
if r2_train < 0.5 and r2_test < 0.5:
    print("Posible underfit: el modelo no está capturando la relación.")
elif abs(r2_train - r2_test) > 0.15 and r2_train > r2_test:
    print("Posible overfit: el modelo se ajusta demasiado al entrenamiento.")
else:
    print("Buen ajuste: el modelo generaliza bien.")

# 6. Diagnóstico de bias y varianza
print("\nDiagnóstico de bias y varianza:")
if mse_train < mse_test and abs(mse_train - mse_test) > 10:
    print("Alta varianza: el modelo podría estar sobreajustado.")
elif mse_train > mse_test:
    print("Posible alta bias: el modelo no aprende bien del entrenamiento.")
else:
    print("Bias y varianza en equilibrio.")

print("\nGráficas guardadas y métricas impresas para tu reporte.")