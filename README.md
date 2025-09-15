# Análisis Comparativo de Modelos de Regresión

Este repositorio contiene la implementación, ajuste y análisis comparativo de modelos de regresión sobre el dataset California Housing. El objetivo es evaluar el desempeño de diferentes algoritmos, diagnosticar bias y varianza, y demostrar el efecto de técnicas de regularización y ajuste de hiperparámetros.

## Estructura del repositorio

```
dataset/
    datos_california.csv
models/
    modelo_decision_tree.joblib
    decision_tree_adjusted.joblib
    elasticnet_california.joblib
resources/
    (aquí se guardan todas las gráficas generadas)
codes/
    models/
        decisionTree_Regressor_framework.py
        decisionTree_Regressor_framework_adjusted.py
        elasticNet_framework.py
        ...
    performanceAnalysis.py
README.md
```

## Flujo de trabajo

1. **Preprocesamiento de datos**
   - Ejecuta el script para generar `dataset/datos_california.csv` usando el dataset California Housing de scikit-learn.

2. **Entrenamiento y guardado de modelos**
   - Entrena y guarda los modelos:
     - Árbol de Decisión (sin ajuste)
     - Árbol de Decisión (con hiperparámetros ajustados)
     - ElasticNet (con regularización)
   - Los modelos se guardan en la carpeta `models/` usando `joblib`.

3. **Análisis de desempeño**
   - Ejecuta `performanceAnalysis.py` para:
     - Cargar los modelos entrenados.
     - Evaluar cada modelo en conjuntos de entrenamiento y prueba.
     - Calcular métricas (MSE, MAE, R²).
     - Diagnosticar bias, varianza y nivel de ajuste (underfit, overfit, buen ajuste).
     - Graficar:
       - Predicción vs Real
       - Histograma de errores
       - Boxplot de métricas
       - Curva de aprendizaje
     - Todas las gráficas se guardan en la carpeta `resources/`.

4. **Reporte**
   - Utiliza las métricas y gráficas generadas para documentar:
     - Comparación de desempeño entre modelos.
     - Diagnóstico de bias y varianza.
     - Efecto de la regularización y el ajuste de hiperparámetros.
     - Explicación del nivel de ajuste de cada modelo.

## Ejecución rápida

1. Preprocesa los datos:
   ```bash
   python codes/preprocess/preprocess_Trial_Data.py
   ```
2. Entrena y guarda los modelos:
   ```bash
   python codes/models/decisionTree_Regressor_framework.py
   python codes/models/decisionTree_Regressor_framework_adjusted.py
   python codes/models/elasticNet_framework.py
   ```
3. Ejecuta el análisis de desempeño:
   ```bash
   python codes/performanceAnalysis.py
   ```

## Requisitos

- Python 3.8+
- numpy
- scikit-learn
- matplotlib
- joblib

Instala las dependencias con:
```bash
pip install numpy scikit-learn matplotlib joblib
```

## Notas

- Puedes agregar o comparar otros modelos siguiendo la misma estructura.
- El análisis y las gráficas generadas cumplen con los criterios de evaluación del curso: separación de conjuntos, diagnóstico de bias/varianza, nivel de ajuste y uso de regularización.
- El reporte final debe incluir las gráficas y explicaciones generadas por `performanceAnalysis.py`.

---

**Autor:** Edwin Iñiguez Moncada  
**Licencia:** MIT