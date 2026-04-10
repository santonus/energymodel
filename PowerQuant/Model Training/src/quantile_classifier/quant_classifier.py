import numpy as np


def encode_to_quantiles(y, ind):
    """Convert values to quantiles within categories."""
    y_quant = np.zeros(len(y))
    for cat in np.unique(ind):
        mask = ind == cat
        values = y[mask]
        counts = np.mean(values[:, None] > values, axis=1)
        y_quant[mask] = counts
    return y_quant


def decode_from_quantiles(quantiles, val):
    """Convert quantiles back to values using category information."""
    return np.quantile(val, quantiles, method='averaged_inverted_cdf')


class QuantileClassifier:
    """Wrapper class that transforms any classifier to predict quantiles within categories."""

    def __init__(self, base_classifier, **kwargs):
        """Initialize with a pre-configured base classifier."""
        self.base_classifier = base_classifier

    def fit(self, X, y, ind):
        """Fit the classifier on quantile-transformed targets."""
        assert ind is not None, 'ind must be provided.'
        assert len(ind) == len(y), 'ind and y must have the same length.'
        y_quant = encode_to_quantiles(y, ind)
        self.base_classifier.fit(X, y_quant)
        self.ind = ind
        self.val = y.copy()

    def predict(self, X, ind=None, val=None):
        """Predict values by converting quantile predictions back to original scale."""
        if ind is None and val is None:
            raise ValueError('Either idx or val must be provided.')

        if ind is not None:
            assert len(ind) == len(X), 'ind and X must have the same length.'
            y_pred = np.zeros(len(X))
            for idx in np.unique(ind):
                mask = ind == idx
                val = self.val[self.ind == idx]
                y_quant_pred = self.base_classifier.predict(X[mask])
                y_quant_pred = np.clip(y_quant_pred, 1e-3, 1 - 1e-3)
                y_pred[mask] = decode_from_quantiles(y_quant_pred, val)
            return y_pred

        if val is not None:
            y_quant_pred = self.base_classifier.predict(X)
            y_quant_pred = np.clip(y_quant_pred, 1e-3, 1 - 1e-3)
            y_pred = decode_from_quantiles(y_quant_pred, val)
            return y_pred
