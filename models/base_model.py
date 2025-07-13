# models/base_model.py

from abc import ABC, abstractmethod
import pandas as pd

class BaseModel(ABC):
    """
    Principio SOLID aplicado: Interface Segregation & Dependency Inversion
    Define una interfaz clara para cualquier modelo ML que se desee usar.
    """

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        pass

    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        pass
