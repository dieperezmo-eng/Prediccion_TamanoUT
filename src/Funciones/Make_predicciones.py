import pandas as pd
import pickle
import gzip
import json
import os

# ----------------------------------------------------------
# 1. Cargar modelo
# ----------------------------------------------------------

def cargar_modelo(ruta_modelo: str):
    """Carga un modelo entrenado desde un archivo .pkl.gz"""
    with gzip.open(ruta_modelo, 'rb') as f:
        modelo = pickle.load(f)
    return modelo

# ----------------------------------------------------------
# 2. Cargar columnas del modelo
# ----------------------------------------------------------

def cargar_columnas_modelo(ruta_columnas: str):
    """Carga las columnas que el modelo espera recibir."""
    with open(ruta_columnas, 'r') as f:
        columnas = json.load(f)
    return columnas

# ----------------------------------------------------------
# 3. Cargar datos nuevos
# ----------------------------------------------------------

def cargar_datos_prediccion(ruta_csv: str, columnas_modelo):
    """
    Carga el CSV con clientes y prepara las columnas necesarias
    Mantiene la columna Nit para el resultado final.
    """

    df = pd.read_csv(ruta_csv).copy()

    # Verificar que exista Nit
    if "nit-razon social" not in df.columns:
        raise ValueError("ERROR: El archivo de predicción NO contiene la columna 'Nit'.")

    if "nit-razon social" not in df.columns:
        raise ValueError("ERROR: No se encontró columna 'Nit' después de normalizar nombres.")

    # Validar existencia de las columnas necesarias
    faltantes = [c for c in columnas_modelo if c not in df.columns]

    if faltantes:
        raise ValueError(f"ERROR: Faltan columnas necesarias para predecir: {faltantes}")

    # Extraer DataFrame solo con las columnas del modelo
    X_pred = df[columnas_modelo].copy()

    return df, X_pred  # df completo y X_pred solo features

# ----------------------------------------------------------
# 4. Hacer predicciones
# ----------------------------------------------------------

def hacer_predicciones(modelo, df_original, X_pred):
    """Genera la predicción y devuelve un DF final incluyendo Nit."""

    df_resultado = df_original.copy()

    df_resultado["prediccion_ut"] = modelo.predict(X_pred)

    return df_resultado

# ----------------------------------------------------------
# 5. Guardar resultados
# ----------------------------------------------------------

def guardar_predicciones(df, ruta_salida: str):
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    df.to_csv(ruta_salida, index=False)
    print(f"✔ Archivo guardado en: {ruta_salida}")

# ----------------------------------------------------------
# 6. MAIN
# ----------------------------------------------------------

if __name__ == "__main__":

    # -----------------------------
    # Rutas
    # -----------------------------
    ruta_modelo = "data/models/random_forest.pkl.gz"            # <-- Cambia aquí el modelo a usar
    ruta_columnas = "data/models/columns_modelo.json"
    ruta_datos = "Data/02-Preprocessed/Dataprediccion.csv"
    ruta_salida = "Data/04-Predictions/predicciones_ut.csv"

    print("Cargando modelo...")
    modelo = cargar_modelo(ruta_modelo)

    print("Cargando columnas del modelo...")
    columnas_modelo = cargar_columnas_modelo(ruta_columnas)

    print("Cargando archivo de predicción...")
    df_original, X_pred = cargar_datos_prediccion(ruta_datos, columnas_modelo)

    print("Haciendo predicciones...")
    df_pred = hacer_predicciones(modelo, df_original, X_pred)

    print("Guardando resultados...")
    guardar_predicciones(df_pred, ruta_salida)

    print("\n✔✔ Predicciones generadas correctamente.")
