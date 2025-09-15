import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, learning_curve
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

# Asegúrate de que la carpeta resources existe
os.makedirs("resources", exist_ok=True)

# Cargar y separar datos
X, y = cargar_datos('dataset/datos_california.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Función para graficar curvas de aprendizaje
def plot_learning_curve(model, X, y, nombre_archivo, titulo):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, scoring='neg_mean_squared_error',
        train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
    )
    train_scores_mean = -np.mean(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)

    plt.figure(figsize=(8,5))
    plt.plot(train_sizes, train_scores_mean, 'o-', label="Entrenamiento")
    plt.plot(train_sizes, test_scores_mean, 'o-', label="Validación")
    plt.xlabel("Tamaño del conjunto de entrenamiento")
    plt.ylabel("MSE")
    plt.title(titulo)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join("resources", nombre_archivo))
    plt.close()

# =========================
# Análisis: DecisionTreeRegressor (sin regularización)
# =========================
modelo_tree = joblib.load('models/modelo_decision_tree.joblib')

y_pred_train_tree = modelo_tree.predict(X_train)
y_pred_test_tree = modelo_tree.predict(X_test)

mse_train_tree = mean_squared_error(y_train, y_pred_train_tree)
mae_train_tree = mean_absolute_error(y_train, y_pred_train_tree)
r2_train_tree = r2_score(y_train, y_pred_train_tree)

mse_test_tree = mean_squared_error(y_test, y_pred_test_tree)
mae_test_tree = mean_absolute_error(y_test, y_pred_test_tree)
r2_test_tree = r2_score(y_test, y_pred_test_tree)

print("Árbol de Decisión (sin regularización) - Entrenamiento:")
print(f"  MSE: {mse_train_tree:.2f} | MAE: {mae_train_tree:.2f} | R2: {r2_train_tree:.2f}")
print("Árbol de Decisión (sin regularización) - Prueba:")
print(f"  MSE: {mse_test_tree:.2f} | MAE: {mae_test_tree:.2f} | R2: {r2_test_tree:.2f}")

# Gráficas DecisionTreeRegressor
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred_test_tree, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Valor real")
plt.ylabel("Predicción")
plt.title("Predicción vs Real (Árbol de Decisión)")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join("resources", "pred_vs_real_decision_tree.png"))
plt.close()

errores_tree = y_test - y_pred_test_tree
plt.figure(figsize=(8,5))
plt.hist(errores_tree, bins=20, color='skyblue', edgecolor='black')
plt.xlabel("Error (residuo)")
plt.ylabel("Frecuencia")
plt.title("Distribución de errores (Árbol de Decisión)")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join("resources", "hist_errores_decision_tree.png"))
plt.close()

plt.figure(figsize=(8,5))
plt.boxplot([[mse_train_tree, mse_test_tree], [mae_train_tree, mae_test_tree], [r2_train_tree, r2_test_tree]],
            labels=["MSE", "MAE", "R2"])
plt.title("Comparación de métricas (Árbol de Decisión)")
plt.ylabel("Valor")
plt.tight_layout()
plt.savefig(os.path.join("resources", "boxplot_metricas_decision_tree.png"))
plt.close()

plot_learning_curve(modelo_tree, X_train, y_train, "learning_curve_decision_tree.png", "Curva de aprendizaje (Árbol de Decisión)")

print("\nDiagnóstico Árbol de Decisión:")
if r2_train_tree < 0.5 and r2_test_tree < 0.5:
    print("Posible underfit: el modelo no está capturando la relación.")
elif abs(r2_train_tree - r2_test_tree) > 0.15 and r2_train_tree > r2_test_tree:
    print("Posible overfit: el modelo se ajusta demasiado al entrenamiento.")
else:
    print("Buen ajuste: el modelo generaliza bien.")

if mse_train_tree < mse_test_tree and abs(mse_train_tree - mse_test_tree) > 10:
    print("Alta varianza: el modelo podría estar sobreajustado.")
elif mse_train_tree > mse_test_tree:
    print("Posible alta bias: el modelo no aprende bien del entrenamiento.")
else:
    print("Bias y varianza en equilibrio.")

# =========================
# Análisis: ElasticNet (con regularización)
# =========================
modelo_enet = joblib.load('models/elasticnet_california.joblib')

y_pred_train_enet = modelo_enet.predict(X_train)
y_pred_test_enet = modelo_enet.predict(X_test)

mse_train_enet = mean_squared_error(y_train, y_pred_train_enet)
mae_train_enet = mean_absolute_error(y_train, y_pred_train_enet)
r2_train_enet = r2_score(y_train, y_pred_train_enet)

mse_test_enet = mean_squared_error(y_test, y_pred_test_enet)
mae_test_enet = mean_absolute_error(y_test, y_pred_test_enet)
r2_test_enet = r2_score(y_test, y_pred_test_enet)

print("\nElasticNet (con regularización) - Entrenamiento:")
print(f"  MSE: {mse_train_enet:.2f} | MAE: {mae_train_enet:.2f} | R2: {r2_train_enet:.2f}")
print("ElasticNet (con regularización) - Prueba:")
print(f"  MSE: {mse_test_enet:.2f} | MAE: {mae_test_enet:.2f} | R2: {r2_test_enet:.2f}")

# Gráficas ElasticNet
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred_test_enet, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Valor real")
plt.ylabel("Predicción")
plt.title("Predicción vs Real (ElasticNet)")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join("resources", "pred_vs_real_elasticnet.png"))
plt.close()

errores_enet = y_test - y_pred_test_enet
plt.figure(figsize=(8,5))
plt.hist(errores_enet, bins=20, color='salmon', edgecolor='black')
plt.xlabel("Error (residuo)")
plt.ylabel("Frecuencia")
plt.title("Distribución de errores (ElasticNet)")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join("resources", "hist_errores_elasticnet.png"))
plt.close()

plt.figure(figsize=(8,5))
plt.boxplot([[mse_train_enet, mse_test_enet], [mae_train_enet, mae_test_enet], [r2_train_enet, r2_test_enet]],
            labels=["MSE", "MAE", "R2"])
plt.title("Comparación de métricas (ElasticNet)")
plt.ylabel("Valor")
plt.tight_layout()
plt.savefig(os.path.join("resources", "boxplot_metricas_elasticnet.png"))
plt.close()

plot_learning_curve(modelo_enet, X_train, y_train, "learning_curve_elasticnet.png", "Curva de aprendizaje (ElasticNet)")

print("\nDiagnóstico ElasticNet:")
if r2_train_enet < 0.5 and r2_test_enet < 0.5:
    print("Posible underfit: el modelo no está capturando la relación.")
elif abs(r2_train_enet - r2_test_enet) > 0.15 and r2_train_enet > r2_test_enet:
    print("Posible overfit: el modelo se ajusta demasiado al entrenamiento.")
else:
    print("Buen ajuste: el modelo generaliza bien.")

if mse_train_enet < mse_test_enet and abs(mse_train_enet - mse_test_enet) > 10:
    print("Alta varianza: el modelo podría estar sobreajustado.")
elif mse_train_enet > mse_test_enet:
    print("Posible alta bias: el modelo no aprende bien del entrenamiento.")
else:
    print("Bias y varianza en equilibrio.")

# =========================
# Análisis: DecisionTreeRegressor (ajustado)
# =========================
modelo_tree_adj = joblib.load('models/decision_tree_adjusted.joblib')

y_pred_train_tree_adj = modelo_tree_adj.predict(X_train)
y_pred_test_tree_adj = modelo_tree_adj.predict(X_test)

mse_train_tree_adj = mean_squared_error(y_train, y_pred_train_tree_adj)
mae_train_tree_adj = mean_absolute_error(y_train, y_pred_train_tree_adj)
r2_train_tree_adj = r2_score(y_train, y_pred_train_tree_adj)

mse_test_tree_adj = mean_squared_error(y_test, y_pred_test_tree_adj)
mae_test_tree_adj = mean_absolute_error(y_test, y_pred_test_tree_adj)
r2_test_tree_adj = r2_score(y_test, y_pred_test_tree_adj)

print("Árbol de Decisión (ajustado) - Entrenamiento:")
print(f"  MSE: {mse_train_tree_adj:.2f} | MAE: {mae_train_tree_adj:.2f} | R2: {r2_train_tree_adj:.2f}")
print("Árbol de Decisión (ajustado) - Prueba:")
print(f"  MSE: {mse_test_tree_adj:.2f} | MAE: {mae_test_tree_adj:.2f} | R2: {r2_test_tree_adj:.2f}")

# Gráficas DecisionTreeRegressor ajustado
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred_test_tree_adj, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Valor real")
plt.ylabel("Predicción")
plt.title("Predicción vs Real (Árbol de Decisión Ajustado)")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join("resources", "pred_vs_real_decision_tree_adjusted.png"))
plt.close()

errores_tree_adj = y_test - y_pred_test_tree_adj
plt.figure(figsize=(8,5))
plt.hist(errores_tree_adj, bins=20, color='skyblue', edgecolor='black')
plt.xlabel("Error (residuo)")
plt.ylabel("Frecuencia")
plt.title("Distribución de errores (Árbol de Decisión Ajustado)")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join("resources", "hist_errores_decision_tree_adjusted.png"))
plt.close()

plt.figure(figsize=(8,5))
plt.boxplot([[mse_train_tree_adj, mse_test_tree_adj], [mae_train_tree_adj, mae_test_tree_adj], [r2_train_tree_adj, r2_test_tree_adj]],
            labels=["MSE", "MAE", "R2"])
plt.title("Comparación de métricas (Árbol de Decisión Ajustado)")
plt.ylabel("Valor")
plt.tight_layout()
plt.savefig(os.path.join("resources", "boxplot_metricas_decision_tree_adjusted.png"))
plt.close()

plot_learning_curve(modelo_tree_adj, X_train, y_train, "learning_curve_decision_tree_adjusted.png", "Curva de aprendizaje (Árbol de Decisión Ajustado)")

print("\nDiagnóstico Árbol de Decisión Ajustado:")
if r2_train_tree_adj < 0.5 and r2_test_tree_adj < 0.5:
    print("Posible underfit: el modelo no está capturando la relación.")
elif abs(r2_train_tree_adj - r2_test_tree_adj) > 0.15 and r2_train_tree_adj > r2_test_tree_adj:
    print("Posible overfit: el modelo se ajusta demasiado al entrenamiento.")
else:
    print("Buen ajuste: el modelo generaliza bien.")

if mse_train_tree_adj < mse_test_tree_adj and abs(mse_train_tree_adj - mse_test_tree_adj) > 10:
    print("Alta varianza: el modelo podría estar sobreajustado.")
elif mse_train_tree_adj > mse_test_tree_adj:
    print("Posible alta bias: el modelo no aprende bien del entrenamiento.")
else:
    print("Bias y varianza en equilibrio.")