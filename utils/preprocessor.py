# utils/preprocessor.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(filepath: str):
    """
    Carga y limpia el dataset desde un archivo CSV.
    Aplica codificación a variables categóricas.
    Devuelve X (features) e y (target).
    """
    df = pd.read_csv(filepath)

    # Convertir fechas a datetime
    df['fecha'] = pd.to_datetime(df['fecha'])
    df['semana'] = df['fecha'].dt.isocalendar().week.astype(int)
    df['anio'] = df['fecha'].dt.year.astype(int)

    

    # Codificación de variables categóricas
    le_prod = LabelEncoder()
    le_tienda = LabelEncoder()
    df['producto_id'] = le_prod.fit_transform(df['producto_id'])
    df['tienda_id'] = le_tienda.fit_transform(df['tienda_id'])

    # Variable objetivo: cantidad vendida
    y = df['cantidad']

    # Variables predictoras
    X = df[['producto_id', 'stock', 'precio_unitario', 'tienda_id', 'semana', 'anio']]

    return X, y
