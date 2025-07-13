# models/random_forest_model.py

from models.base_model import BaseModel
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib
import os

class RandomForestModel(BaseModel):
    """
    Modelo basado en Random Forest para predicción de demanda.
    Aplica principio SOLID: puede reemplazar a cualquier modelo sin modificar el servicio.
    """

    def __init__(self, model_path='random_forest_model.pkl'):
        self.model_path = model_path
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
            except Exception as e:
                print("Error cargando modelo Random Forest:", e)

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)
        joblib.dump(self.model, self.model_path)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model.predict(X), index=X.index)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        preds = self.predict(X)
        mae = mean_absolute_error(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))
        return {"MAE": mae, "RMSE": rmse}
