import numpy as np
from .utils import *


class DecisionStump:
    """
    Реализация слабого алгоритма - решающего пня
    """

    def __init__(self):
        """
        Инициализация слабого алгоритма, решающего пня

        Parameters:
        ----------
        feature_index: индекс признака, по которому строится порог, по умолчанию None
        threshold: порог, по которому разбивается признак, по умолчанию None
        polarity: направление разбиения, по умолчанию 1
        """
        self.feature_index = None
        self.threshold = None
        self.polarity = 1

    def fit(self, X, y, w=None):
        """
        Обучение слабого алгоритма, решающего пня

        Parameters:
        ----------
        X: матрица признаков
        y: вектор ответов
        w: вектор весов, по умолчанию None
        """
        X, y, w = check_data(X, y, w)

        n_samples, n_features = X.shape

        if w is None:
            w = np.array([1.0 / n_samples for _ in range(n_samples)])

        best_err = float("inf")

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
        Предсказание слабого алгоритма, решающего пня

        Parameters:
        ----------
        X: матрица признаков

        Returns:
        ----------
        y_pred: вектор предсказаний
        """
        if self.feature_index is None or self.threshold is None:
            raise ValueError("DecisionStump must be fitted before prediction")

        if self.polarity == 1:
            return (X[:, self.feature_index] >= self.threshold).astype(int)
        else:
            return (X[:, self.feature_index] < self.threshold).astype(int)


class Node:
    def __init__(
        self, feature_index=None, threshold=None, left=None, right=None, value=None
    ):
        """
        Реализация вершины дерева для слабого алгоритма, решающего дерева

        Parameters:
        ----------
        feature_index: индекс признака для разбиения, по умолчанию None
        threshold: порог признака для разбиения, по умолчанию None
        left: левое поддерево, по умолчанию None
        right: правое поддерево, по умолчанию None
        value: значение класса для листа, по умолчанию None
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.is_leaf = True if value is not None else False

    def predict(self, x):
        """
        Предсказания для одного объекта

        Parameters:
        ----------
        x: вектор признаков объекта

        Returns:
        ----------
        prediction: предсказание
        """

        if self.is_leaf:
            return self.value

        if x[self.feature_index] < self.threshold:
            return self.left.predict(x)
        else:
            return self.right.predict(x)


class DecisionTree:
    """
    Реализация слабого алгоритма, решающего дерева
    """

    def __init__(self, max_depth=3, min_samples_split=2):
        """
        Инициализация слабого алгоритма, решающего дерева

        Parameters:
        ----------
        max_depth: максимальная глубина дерева, по умолчанию 3
        min_samples_split: минимальное количество объектов для разбиения, по умолчанию 2
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y, w=None):
        """
        Обучение слабого алгоритма, решающего дерева

        Parameters:
        ----------
        X: матрица признаков
        y: вектор ответов
        w: вектор весов, по умолчанию None
        """
        X, y, w = check_data(X, y, w)
        n_samples = X.shape[0]

        if w is None:
            w = np.array([1.0 / n_samples for _ in range(n_samples)])

        self.root = self._build_tree(X, y, w, depth=0)

    def _build_tree(self, X, y, w, depth):
        """
        Рекурсивное построение решающего дерева

        Parameters:
        ----------
        X: матрица признаков
        y: вектор ответов
        w: вектор весов
        depth: текущая глубина

        Returns:
        ----------
        node: вершину дерева
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or n_classes == 1
        ):
            leaf_value = self._most_significant_class(y, w)
            return Node(value=leaf_value)

        best_feature_index, best_threshold = self._best_split(X, y, w, n_features)

        if best_feature_index is None:
            leaf_value = self._most_significant_class(y, w)
            return Node(value=leaf_value)

        left_indxs = X[:, best_feature_index] < best_threshold
        right_indxs = X[:, best_feature_index] >= best_threshold

        left_child = self._build_tree(
            X[left_indxs], y[left_indxs], w[left_indxs], depth + 1
        )
        right_child = self._build_tree(
            X[right_indxs], y[right_indxs], w[right_indxs], depth + 1
        )

        return Node(
            feature_index=best_feature_index,
            threshold=best_threshold,
            left=left_child,
            right=right_child,
        )

    def _best_split(self, X, y, w, n_features):
        """
        Поиск лучшего разбиения

        Parameters:
        ----------
        X: матрица признаков
        y: вектор ответов
        w: вектор весов
        n_features: количество признаков

        Returns:
        ----------
        best_feature: индекс лучшего признака
        best_threshold: лучший порог
        """
        best_err = float("inf")
        best_feature_index = None
        best_threshold = None

        for feature_index in range(n_features):

            thresholds = np.unique(X[:, feature_index])

            for thr in thresholds:
                left_indxs = X[:, feature_index] < thr
                right_indxs = X[:, feature_index] >= thr

                if np.sum(left_indxs) == 0 or np.sum(right_indxs) == 0:
                    continue

                left_pred = self._most_significant_class(y[left_indxs], w[left_indxs])
                right_pred = self._most_significant_class(
                    y[right_indxs], w[right_indxs]
                )

                y_pred = np.zeros_like(y)
                y_pred[left_indxs] = left_pred
                y_pred[right_indxs] = right_pred

                err = calc_error(y, y_pred, w)

                if err < best_err:
                    best_err = err
                    best_feature_index = feature_index
                    best_threshold = thr

        return best_feature_index, best_threshold

    def _most_significant_class(self, y, w):
        """
        Поиск наиболее весомого класса

        Parameters:
        ----------
        y: вектор ответов
        w: вектор весов

        Returns:
        ----------
        mx_class: наиболее весомый класс
        """
        classes = np.unique(y)
        mx = 0
        mx_class = None

        for c in classes:
            cnt = np.sum(w[y == c])

            if cnt > mx:
                mx_class = c
                mx = cnt

        return mx_class

    def predict(self, X):
        """
        Предсказание слабого алгоритма, решающего дерева

        Parameters:
        ----------
        X: матрица признаков

        Returns:
        ----------
        y_pred: вектор предсказаний
        """
        if self.root is None:
            raise ValueError("DecisionTree must be fitted before prediction")

        return np.array([self.root.predict(x) for x in X])
