import numpy as np

def check_X(X):
    """
    Проверка входных данных матрицы объектов: непустота, валидная размерность

    Parameters:
    ----------
    X : матрица признаков

    Returns:
    X : матрица признаков
    """

    X = np.array(X)

    if len(X) == 0:
        raise ValueError("Data must be non-empty")

    if X.shape[1] == 0:
        raise ValueError("X must have at least one feature")
    
    if X.ndim != 2:
        raise ValueError("X must be 2D (n_samples, n_features)")

    return X

def check_data(X, y=None, p=None):
    """
    Проверка входных данных: непустота, одинаковая размерность, принадлежность симплексу

    Parameters:
    ----------
    X : матрица признаков
    y : вектор ответов
    p : вектор уверенностей

    Returns:
    X : матрица признаков
    y: вектор ответов
    p : вектор уверенностей
    """

    X = np.array(X)
    y = np.array(y)

    if p is not None:
        p = np.array(p, dtype=float)
        check_norm(p)

    if len(X) == 0 or len(y) == 0 or (p is not None and len(p) == 0):
        raise ValueError("Data must be non-empty")

    if X.shape[1] == 0:
        raise ValueError("X must have at least one feature")
    
    if X.ndim != 2:
        raise ValueError("X must be 2D (n_samples, n_features)")
    if y.ndim != 1:
        raise ValueError("y must be 1D (n_samples,)")
    if p is not None:
        if p.ndim != 1:
            raise ValueError("p must be 1D (n_samples,)")
        if p.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in p and y must be equal")

    if X.shape[0] != y.shape[0]:
        raise ValueError("Number of samples in X and y must be equal")

    check_classes(y)

    if p is None:
        p = np.array([1.0 / len(y) for _ in range(len(y))], dtype=float)

    return X, y, p


def check_classes(y):
    """
    Проверка классов: должно быть 2 класса

    Parameters:
    ----------
    y : вектор ответов
    """
    if len(np.unique(y)) != 2:
        raise ValueError("Number of classes must be equal to 2")


def check_norm(p, ep=1e-8):
    """
    Проверка принадлежности вектора уверенностей симплексу: сумма компонентов должна быть равна 1
    а все компоненты должны быть неотрицательны

    Parameters:
    ----------
    p : вектор уверенностей
    ep : точность проверки суммы компонентов вектора уверенностей
    """
    if not np.isclose(p.sum(), 1.0, atol=ep) or np.any(p < -ep):
        raise ValueError("p must be in simplex")


def check_error(eps):
    """
    Проверка ошибки на принадлежность интервалу (0, 0.5)
    """
    if eps <= 0 or eps >= 0.5:
        raise ValueError("Error must be in interval (0, 0.5)")


def calc_error(y, y_pred, p=None):
    """
    Вычисление ошибки

    Parameters:
    ----------
    y : вектор ответов
    y_pred : вектор предсказаний
    p : вектор уверенностей

    Returns:
    ----------
    eps : ошибка
    """
    # вычисляем ошибку
    if p is None:
        p = np.array([1.0 / len(y) for i in range(len(y))], dtype=float)

    eps = sum(p * (y != y_pred))
    
    return eps


def update_weight(y, y_pred, w, beta):
    """
    Обновление весов

    Parameters:
    ----------
    y : вектор ответов
    y_pred : вектор предсказаний
    w : вектор весов
    beta : коэффициент поправки бета

    Returns:
    ----------
    p : нормированный вектор уверенностей
    """
    w = w * (beta ** (y == y_pred).astype(int))
    p = w / w.sum()
    check_norm(p)

    return p

def calc_accuracy(y, y_pred):
    """
    Вычисление метрики accuracy

    Parameters:
    ----------
    y : вектор ответов
    y_pred : вектор предсказаний
    """
    y = np.array(y)
    y_pred = np.array(y_pred)

    return (y == y_pred).mean()