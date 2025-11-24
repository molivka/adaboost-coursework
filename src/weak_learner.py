import numpy as np
from .utils import *


class DecisionStump:
    """
    Реализация слабого алгоритма - решающего пня
    """

    def __init__(self):
        """
        Инициализация слабого алгоритма

        Parameters:
        ----------
        feature_index : индекс признака, по которому строится порог
        threshold : порог, по которому разбивается признак
        polarity : направление разбиения
        """
        self.feature_index = None
        self.threshold = None
        self.polarity = 1
    
    def fit(self, X, y, w=None):
        """
        Обучение слабого алгоритма

        Parameters:
        ----------
        X : матрица признаков
        y : вектор ответов
        w : вектор весов
        """
        X, y, w = check_data(X, y, w)
        n_samples, n_features = X.shape
        if w is None:
            w = np.array([1.0 / n_samples for _ in range(n_samples)])
        best_err = float('inf')
        for feature_index in range(n_features):
            threshlods = np.unique(X[:, feature_index])
            for thr in threshlods:
                y_pred = (X[:, feature_index] >= thr).astype(int)
                err = calc_error(y, y_pred, w)
                if err < best_err:
                    best_err = err
                    self.feature_index = feature_index
                    self.threshold = thr
                    self.polarity = 1

                y_pred_inv = (X[:, feature_index] < thr).astype(int)
                err_inv = calc_error(y, y_pred_inv, w)
                if err_inv < best_err:
                    best_err = err_inv
                    self.feature_index = feature_index
                    self.threshold = thr
                    self.polarity = -1

    def predict(self, X):
        """
        Предсказание слабого алгоритма

        Parameters:
        ----------
        X : матрица признаков

        Returns:
        ----------
        y_pred : вектор предсказаний
        """
        if self.feature_index is None or self.threshold is None:
            raise ValueError("DecisionStump must be fitted before prediction")

        if self.polarity == 1:
            return (X[:, self.feature_index] >= self.threshold).astype(int)
        else:
            return (X[:, self.feature_index] < self.threshold).astype(int)

