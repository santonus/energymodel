from typing import Any, Dict

from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor


class RandomForestConfig(BaseModel):
    n_estimators: int = 100
    max_depth: int = 10
    max_features: float = 1.0
    min_samples_leaf: int = 10
    bootstrap: bool = True
    random_state: int = 42
    verbose: int = 100

    def to_json_dict(self) -> Dict[str, Any]:
        """Convert config to JSON-serializable dictionary."""
        data = self.model_dump()
        return data


class RandomForest:
    def __init__(self, config: RandomForestConfig):
        self.model = RandomForestRegressor(**config.model_dump())

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)

    def get_params(self, **kwargs):
        return self.model.get_params(**kwargs)

    def set_params(self, **kwargs):
        self.model.set_params(**kwargs)
