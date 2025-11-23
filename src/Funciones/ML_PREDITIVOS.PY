import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from scipy.stats import randint
import pickle
import gzip
import json
import os

# ----------------------------------------------------------
# 1. Cargar y limpiar
# ----------------------------------------------------------

Ruta_df = "Data/02-Preprocessed/Datamodelar.csv"

def load_clean_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path).copy()
    df = df.drop(columns=['fuente', 'nit', 'nit-razon social'], errors='ignore')
    return df

# ----------------------------------------------------------
# 2. Crear pipeline genérico
# ----------------------------------------------------------

def crear_pipeline(modelo):

    categorical_features = ['Sector']
    numerical_features = ['ventas']

    preprocesador = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', RobustScaler(), numerical_features)
        ],
        remainder='drop'
    )

    selector = SelectKBest(score_func=f_regression)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocesador),
        ('feature_selection', selector),
        ('model', modelo)
    ])

    return pipeline

# ----------------------------------------------------------
# 3. Optimizar hiperparámetros
# ----------------------------------------------------------

def hiperparametros(*, modelo, n_divisiones, x_entrenamiento, y_entrenamiento, puntuacion):

    total_features = x_entrenamiento.shape[1]

    if total_features <= 3:
        param_dist = {'feature_selection__k': [1, 2, total_features]}
    else:
        max_k = min(20, total_features)
        param_dist = {'feature_selection__k': randint(2, max_k + 1)}

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

def metrics(modelo, X_train, y_train):

    y_pred_train = modelo.predict(X_train)

    return {
        "train": {
            'r2': r2_score(y_train, y_pred_train),
            'mse': mean_squared_error(y_train, y_pred_train),
            'mad': median_absolute_error(y_train, y_pred_train)
        }
    }

# ----------------------------------------------------------
# 5. Guardar modelo
# ----------------------------------------------------------

def guardar_modelo(modelo, nombre):
    os.makedirs('data/models', exist_ok=True)
    ruta = f'data/models/{nombre}.pkl.gz'

    with gzip.open(ruta, 'wb') as f:
        pickle.dump(modelo, f)

# ----------------------------------------------------------
# 6. Guardar métricas
# ----------------------------------------------------------

def guardar_metricas(metricas, nombre):

    os.makedirs('data/output', exist_ok=True)

    ruta = f"data/output/metrics_{nombre}.json"

    with open(ruta, "w") as f:
        f.write(json.dumps(metricas, indent=4))

# ----------------------------------------------------------
# 7. MAIN — Entrenar los 3 modelos
# ----------------------------------------------------------

if __name__ == '__main__':

    print("Cargando y limpiando datos...")
    df = load_clean_data(Ruta_df)

    print("Preparando X y y...")
    X = df[['Sector', 'ventas']]
    y = df['ut']

    # ----------------------------------------------------
    # Guardar columnas usadas en el modelo
    # ----------------------------------------------------
    os.makedirs("data/models", exist_ok=True)

    columnas_modelo = list(X.columns)

    with open("data/models/columns_modelo.json", "w") as f:
        json.dump(columnas_modelo, f, indent=4)

    print("✔ Columnas del modelo guardadas en data/models/columns_modelo.json")

    modelos_dict = {
        "random_forest": RandomForestRegressor(),
        "gradient_boosting": GradientBoostingRegressor(),
        "xgboost": XGBRegressor(
            objective='reg:squarederror',
            eval_metric='mae',
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6
        )
    }

    for nombre, modelo_base in modelos_dict.items():

        print(f"\n==============================")
        print(f"Entrenando modelo: {nombre}")
        print(f"==============================")

        pipeline = crear_pipeline(modelo_base)

        modelo_opt = hiperparametros(
            modelo=pipeline,
            n_divisiones=5,
            x_entrenamiento=X,
            y_entrenamiento=y,
            puntuacion='neg_mean_absolute_error'
        )

        print("Guardando modelo...")
        guardar_modelo(modelo_opt, nombre)

        print("Calculando métricas...")
        metricas_finales = metrics(modelo_opt, X, y)

        print("Guardando métricas...")
        guardar_metricas(metricas_finales, nombre)

        print(f"✔ Modelo {nombre} entrenado y guardado correctamente")

    print("\n✔✔ Todos los modelos finalizados correctamente.")
