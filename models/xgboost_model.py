# models/xgboost_model.py

from models.base_model import BaseModel
from xgboost import XGBRegressor
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib
import os

class XGBoostModel(BaseModel):
    def __init__(self, model_path='xgboost_model.pkl'):
        self.model_path = model_path
        self.model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
            except Exception as e:
                print("Error loading model:", e)

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
