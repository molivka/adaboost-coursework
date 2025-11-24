import numpy as np

from src.classifier import AdaBoost
from src.weak_learner import DecisionStump
from src.utils import calc_accuracy


def make_dataset(n_samples=500, random_state=42):
    """
    Генерация датасета: точки в на плоскости

    Parameters:
    ----------
    n_samples : количество точек в датасете
    random_state : seed для генератора случайных чисел

    Returns:
    ----------
    X : матрица признаков
    y : вектор ответов
    """

    rng = np.random.default_rng(random_state)
    n0 = n_samples // 2
    n1 = n_samples - n0

    X0 = rng.normal(loc=[0.0, 0.0], scale=0.7, size=(n0, 2))
    X1 = rng.normal(loc=[2.0, 2.0], scale=0.7, size=(n1, 2))

    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])

    return X, y


def example_1():
    """
    Пример работы AdaBoost без весов и простым классификатором в виде решающего пня
    """
    X, y = make_dataset()
    print("Example 1:")
    for T in range(1, 100):
        model = AdaBoost(weak_learner=DecisionStump, T=T)
        model.fit(X, y)

        y_pred = model.predict(X)
        accuracy = calc_accuracy(y, y_pred)
        print(f"T: {T}, accuracy: {accuracy}")


def example_2():
    """
    Пример работы AdaBoost с весами и простым классификатором в виде решающего пня
    """
    X, y = make_dataset()

    p = np.ones(len(y), dtype=float)
    p[y == 1] = 2.0
    p = p / p.sum() 
    
    print("Example 2:")
    for T in range(1, 100):
        model = AdaBoost(weak_learner=DecisionStump, T=T)
        model.fit(X, y, p=p)

        y_pred = model.predict(X)
        accuracy = calc_accuracy(y, y_pred)
        print(f"T: {T}, accuracy: {accuracy}")


if __name__ == "__main__":
    example_1()
    example_2()
