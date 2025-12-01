import numpy as np
from .utils import *
from .weak_learner import DecisionStump


class AdaBoost:
    """
    Реализация алгоритма AdaBoost с адаптивным взвешиванием
    """

    def __init__(self, weak_learner=DecisionStump, T=100):
        """
        Инициализация класса AdaBoost

        Parameters:
        ----------
        weak_learner: класс слабого классификатора, по умолчанию DecisionStump
        T: кол-во шагов, по умолчанию 100
        """
        self.weak_learner = weak_learner
        self.T = T
        self.hs = []
        self.betas = []
        self.ensemble_error = []
        self.steps = []
        self.errors = []
        self.upper_bounds = []
        self.weights = []

    def fit(self, X, y, p=None, visualize=False):
        """
        Обучение ансамбля

        Parameters:
        ----------
        X: матрица признаков
        y: вектор ответов
        p: вектор уверенностей, по умолчанию None
        visualize: флаг визуализировать ли график зависиости ошибки на одном алгоритме при исправленных весах,
        ошибки ансамбля и верхней границы ошибки от количества шагов, по умолчанию False
        """
        X, y, p = check_data(X, y, p)
        w = p.copy()

        for t in range(1, self.T + 1):
            h = self.weak_learner()
            h.fit(X, y, w)
            y_pred = h.predict(X)

            eps = calc_error(y, y_pred, w)
            check_error(eps)
            beta = eps / (1 - eps)

            self.betas.append(beta)
            self.hs.append(h)

            if visualize:
                self.steps.append(t)
                self.upper_bounds.append(self.get_upper_bound_of_error())
                self.errors.append(eps)
                self.ensemble_error.append(calc_error(y, self.predict(X)))
                self.weights.append(w.copy())

            w = update_weight(y, y_pred, w, beta)

        if visualize:
            visualize_errors(
                self.errors, self.upper_bounds, self.ensemble_error, self.steps
            )
            visualize_weights(self.weights)

    def predict(self, X):
        """
        Предсказание ансамбля

        Parameters:
        ----------
        X: матрица признаков

        Returns:
        ----------
        y_pred: вектор предсказаний
        """
        X = check_X(X)
        T = len(self.betas)

        if T == 0:
            raise ValueError("AdaBoost must be fitted before predict")

        q = [np.log(1 / self.betas[i]) for i in range(T)]
        q_sum = sum(q)

        if q_sum == 0:
            q = [1.0 / T for _ in range(T)]
        else:
            q = [qi / q_sum for qi in q]

        preds = [h.predict(X) for h in self.hs]

        answer = (sum([q[i] * preds[i] for i in range(T)]) >= 0.5).astype(int)

        return answer

    def get_upper_bound_of_error(self):
        """
        Получение верхней границы ошибки результирующего классификатора по теореме 4

        Returns:
        ----------
        upper_bound: верхняя граница ошибки результирующего классификатора
        """
        errors_array = np.array(self.errors)
        upper_bound = 2 ** len(errors_array) * np.prod(
            np.sqrt(errors_array * (1 - errors_array))
        )

        return upper_bound
