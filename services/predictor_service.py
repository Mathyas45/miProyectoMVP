# services/predictor_service.py

from factories.model_factory import ModelFactory
from utils.preprocessor import load_and_preprocess
from utils.history_logger import save_metrics

class PredictorService:
    """
    Orquesta la lógica de entrenamiento y predicción.
    Cumple con el principio SRP y ahora respeta OCP al usar una fábrica de modelos.
    """

    def __init__(self, data_path: str, model_name: str = "xgboost"):
        self.data_path = data_path
        self.model = ModelFactory.create(model_name)


    def run_training(self):
        X, y = load_and_preprocess(self.data_path)
        self.model.train(X, y)
        metrics = self.model.evaluate(X, y)
        save_metrics(metrics)  # 🔁 guardar métrica con fecha
        return metrics

    def predict_from_file(self):
        X, y = load_and_preprocess(self.data_path)
        return self.model.predict(X)
