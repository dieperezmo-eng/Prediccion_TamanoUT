import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error
import pickle
import gzip
import json
import os
from sklearn.ensemble import RandomForestRegressor



# ----------------------------------------------------------
# 1. Cargar y limpiar
# ----------------------------------------------------------

Ruta_df = "Data/02-Preprocessed/Datamodelar.csv"

def load_clean_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path).copy()
    df = df.drop(columns=['fuente', 'nit', 'nit-razon social'], errors='ignore')
    return df


# ----------------------------------------------------------
# 2. Crear pipeline
# ----------------------------------------------------------

def modelo_regresion_lineal(df: pd.DataFrame) -> Pipeline:

    # columnas existentes en X
    categorical_features = ['Sector']
    numerical_features = ['ventas']   # ut NO va aquí porque es la variable objetivo

    preprocesador = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', MinMaxScaler(), numerical_features)
        ],
        remainder='drop'
    )

    selector = SelectKBest(score_func=f_regression)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocesador),
        ('feature_selection', selector),
        ('regression', RandomForestRegressor())
    ])

    return pipeline


# ----------------------------------------------------------
# 3. Optimizar hiperparámetros
# ----------------------------------------------------------

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

def hiperparametros(
        *,
        modelo,
        n_divisiones,
        x_entrenamiento,
        y_entrenamiento,
        puntuacion
):

    # Número total de columnas disponibles
    total_features = x_entrenamiento.shape[1]

    # evitar rango inválido
    if total_features <= 3:
        # caso extremo: pocos features ⇒ búsqueda mínima
        param_dist = {
            'feature_selection__k': [1, 2, total_features]
        }
    else:
        # rango razonable y siempre válido
        max_k = min(20, total_features)
        param_dist = {
            'feature_selection__k': randint(2, max_k + 1)  # randint(low, high) → high es exclusivo
        }

    estimator = RandomizedSearchCV(
        estimator=modelo,
        param_distributions=param_dist,
        n_iter=10,
        cv=n_divisiones,
        refit=True,
        scoring=puntuacion,
        random_state=42,
        n_jobs=-1
    )

    estimator.fit(x_entrenamiento, y_entrenamiento)

    return estimator



# ----------------------------------------------------------
# 4. Métricas
# ----------------------------------------------------------

def metrics(modelo, X_train, y_train, X_test, y_test):

    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)

    return {
        "train": {
            'r2': r2_score(y_train, y_pred_train),
            'mse': mean_squared_error(y_train, y_pred_train),
            'mad': median_absolute_error(y_train, y_pred_train)
        },
        "test": {
            'r2': r2_score(y_test, y_pred_test),
            'mse': mean_squared_error(y_test, y_pred_test),
            'mad': median_absolute_error(y_test, y_pred_test)
        }
    }


# ----------------------------------------------------------
# 5. Guardar modelo
# ----------------------------------------------------------

def guardar_modelo(modelo):

    os.makedirs('data/models', exist_ok=True)
    with gzip.open('data/models/model.pkl.gz', 'wb') as f:
        pickle.dump(modelo, f)


# ----------------------------------------------------------
# 6. Guardar métricas
# ----------------------------------------------------------

def guardar_metricas(metricas):

    os.makedirs('data/output', exist_ok=True)

    with open("data/output/metrics.json", "w") as f:
        f.write(json.dumps(metricas, indent=4))


# ----------------------------------------------------------
# 7. MAIN
# ----------------------------------------------------------

if __name__ == '__main__':
    print("Cargando y limpiando datos...")
    df = load_clean_data(Ruta_df)

    print("Preparando X y y...")
    X = df[['Sector', 'ventas']]   # SOLO variables explicativas
    y = df['ut']

    print("Creando pipeline...")
    pipeline_modelo = modelo_regresion_lineal(df)

    print("Optimizando hiperparámetros...")
    pipeline_opt = hiperparametros(
        modelo=pipeline_modelo,
        n_divisiones=5,
        x_entrenamiento=X,
        y_entrenamiento=y,
        puntuacion='neg_mean_absolute_error'
    )

    print("Guardando modelo...")
    guardar_modelo(pipeline_opt)

    print("Calculando métricas...")
    metricas_finales = metrics(
        pipeline_opt,
        X, y,
        X, y  # no hay test
    )

    print("Guardando métricas...")
    guardar_metricas(metricas_finales)

    print("✔ Modelo entrenado correctamente")
