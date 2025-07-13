# factories/model_factory.py

from models.xgboost_model import XGBoostModel
from models.random_forest_model import RandomForestModel

class ModelFactory:
    @staticmethod
    def create(model_name: str):
        if model_name == "xgboost":
            return XGBoostModel()
        elif model_name == "random_forest":
            return RandomForestModel()
        else:
            raise ValueError(f"Modelo no soportado: {model_name}")
