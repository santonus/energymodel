from typing import Any, Dict

from catboost import CatBoostRegressor
from pydantic import BaseModel


class CatBoostConfig(BaseModel):
    depth: int = 6
    learning_rate: float = 0.1
    iterations: int = 100
    random_state: int = 42
    verbose: int = 500
    min_data_in_leaf: int = 30
    l2_leaf_reg: float = 3.0
    loss_function: str = 'RMSE'
    eval_metric: str = 'RMSE'

    def to_json_dict(self) -> Dict[str, Any]:
        """Convert config to JSON-serializable dictionary."""
        data = self.model_dump()
        # Convert non-serializable objects to their string representation
        if not isinstance(data['loss_function'], str):
            data['loss_function'] = data['loss_function'].__class__.__name__
        return data


class CatBoost:
    def __init__(self, config: CatBoostConfig):
        self.model = CatBoostRegressor(**config.model_dump())

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)

    def get_params(self, **kwargs):
        return self.model.get_params(**kwargs)

    def set_params(self, **kwargs):
        self.model.set_params(**kwargs)

    def get_feature_importances(self, **kwargs):
        return self.model.get_feature_importances(**kwargs)
