from typing import Any, Dict, Union

import numpy as np
from pydantic import BaseModel
from sklearn.svm import SVR


class SVRConfig(BaseModel):
    kernel: str = 'rbf'  # 'linear', 'poly', 'rbf', 'sigmoid'
    C: float = 1.0
    epsilon: float = 0.1
    gamma: Union[str, float] = 'scale'  # 'scale', 'auto', or float
    degree: int = 3
    coef0: float = 0.0
    verbose: bool = True

    def to_json_dict(self) -> Dict[str, Any]:
        """Convert config to JSON-serializable dictionary."""
        data = self.model_dump()
        return data


class SupportVectorRegressor:
    def __init__(self, config: SVRConfig):
        self.model = SVR(**config.model_dump())

        self.mean_X = None
        self.std_X = None
        self.mean_y = None
        self.std_y = None

    def preprocess_data(self, X, y):
        self.mean_X = np.mean(X, axis=0, keepdims=True)
        self.std_X = np.std(X, axis=0, keepdims=True)
        X = (X - self.mean_X) / self.std_X
        self.mean_y = np.mean(y, axis=0, keepdims=True)
        self.std_y = np.std(y, axis=0, keepdims=True)
        y = (y - self.mean_y) / self.std_y
        return X, y

    def fit(self, X, y, **kwargs):
        X, y = self.preprocess_data(X, y)
        self.model.fit(X, y)

    def predict(self, X):
        X = (X - self.mean_X) / self.std_X
        y_pred = self.model.predict(X)
        y_pred = y_pred * self.std_y + self.mean_y
        return y_pred

    def get_params(self, **kwargs):
        return self.model.get_params(**kwargs)

    def set_params(self, **kwargs):
        self.model.set_params(**kwargs)
