import os
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import glob


def load_Rawdata(file_path: str) -> pd.DataFrame:
    """Carga un archivo CSV y devuelve un DataFrame de pandas."""
    """Load text files in 'input_directory/'"""
    #
    # Lea los archivos de texto en la carpeta input/ y almacene el contenido en
    # un DataFrame de Pandas. Cada línea del archivo de texto debe ser una
    # entrada en el DataFrame.
    #
    files = glob.glob(f"{file_path}/*")
    dataframes = [
        pd.read_csv(file, encoding="utf-8") for file in files
    ]                                                
    dataframe = pd.concat(dataframes, ignore_index=True)
    return dataframe

    #dataframe = load_Rawdata("Data/01-raw")
    #print(dataframe.head(10))  # Mostrar las primeras 10 filas del DataFrame carg