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
        weak_learner : класс слабого классификатора
        T : кол-во шагов
        """
        self.weak_learner = weak_learner
        self.T = T
        self.hs = []
        self.betas = []

    def fit(self, X, y, p=None):
        """
        Обучение ансамбля

        Parameters:
        ----------
        X : матрица признаков
        y : вектор ответов
        p : вектор уверенностей
        """
        X, y, p = check_data(X, y, p)
        w = p.copy()

        for _ in range(self.T):
            h = self.weak_learner()
            h.fit(X, y, w)
            y_pred = h.predict(X)
            eps = calc_error(y, y_pred, w)
            check_error(eps)
            beta = eps / (1 - eps)
            w = update_weight(y, y_pred, w, beta)
            self.betas.append(beta)
            self.hs.append(h)

    def predict(self, X):
        """
        Предсказание ансамбля

        Parameters:
        ----------
        X : матрица признаков

        Returns:
        ----------
        y_pred : вектор предсказаний
        """
        X = check_X(X)
        T = len(self.betas)
        if T == 0:
            raise ValueError("AdaBoost must be fitted before predict.")

        q = [np.log(1 / self.betas[i]) for i in range(T)]
        q_sum = sum(q)
        if q_sum == 0:
            q = [1.0 / T for _ in range(T)]
        else:
            q = [qi / q_sum for qi in q]

        preds = [h.predict(X) for h in self.hs]

        return (sum([q[i] * preds[i] for i in range(self.T)]) >= 0.5).astype(int)