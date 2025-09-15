# Elastic Net Regression: Manual vs Framework

Este repositorio contiene la implementación y comparación de un modelo de regresión Elastic Net, tanto **manual** (sin frameworks de machine learning) como usando **scikit-learn**. El objetivo es demostrar el entendimiento de los algoritmos, el preprocesamiento y la evaluación de modelos en un flujo reproducible.

## Estructura del repositorio

```
dataset/
    datos_regresion.csv
    datos_regresion_escalado.csv
    archive/
        Base.csv
        Variant I.csv
        ...
codes/
    models/
        elasticNet_manual.py
        elasticNet_framework.py
    preprocess/
        preprocess_Trial_Data.py
README.md
```

## Descripción de los archivos principales

- **dataset/**: Contiene los archivos de datos utilizados para entrenar y evaluar los modelos.
- **codes/models/elasticNet_manual.py**: Implementación manual de regresión Elastic Net (sin frameworks).
- **codes/models/elasticNet_framework.py**: Implementación de Elastic Net usando scikit-learn y pipeline.
- **codes/preprocess/preprocess_Trial_Data.py**: Script para cargar, escalar y guardar el dataset de diabetes listo para los modelos.

## Flujo de trabajo

1. **Preprocesamiento de datos**  
   Ejecuta `codes/preprocess/preprocess_Trial_Data.py` para generar el archivo `datos_regresion.csv` a partir del dataset de diabetes de scikit-learn. El script también puede escalar los datos si es necesario.

2. **Entrenamiento y evaluación de modelos**
   - **Manual:** Ejecuta `codes/models/elasticNet_manual.py` para entrenar y evaluar el modelo Elastic Net implementado desde cero.
   - **Framework:** Ejecuta `codes/models/elasticNet_framework.py` para entrenar y evaluar el modelo usando scikit-learn y pipeline.

3. **Comparación y análisis**
   - Ambos scripts reportan métricas de desempeño (MSE, MAE, R²) y realizan validación cruzada k-fold.
   - Puedes comparar los resultados para analizar bias, varianza y el efecto de la regularización.

## Requisitos

- Python 3.8+
- numpy
- scikit-learn

Instala las dependencias con:

```bash
pip install numpy scikit-learn
```

## Ejecución

1. Preprocesa los datos:
   ```bash
   python codes/preprocess/preprocess_Trial_Data.py
   ```
2. Ejecuta el modelo manual:
   ```bash
   python codes/models/elasticNet_manual.py
   ```
3. Ejecuta el modelo con framework:
   ```bash
   python codes/models/elasticNet_framework.py
   ```

## Criterios de evaluación cubiertos

- Separación y evaluación con conjuntos de entrenamiento, validación y prueba.
- Diagnóstico de bias y varianza.
- Explicación del ajuste del modelo (underfit, fit, overfit).
- Uso de técnicas de regularización para mejorar el desempeño.
- Comparación entre implementación manual y con framework.

## Notas

- Puedes modificar los hiperparámetros en los scripts para experimentar con el ajuste y la regularización.
- El reporte detallado y los gráficos comparativos se encuentran en el documento de análisis.

---

**Autor:** Ediwn Iñiguez Moncada  
**Licencia:** MIT