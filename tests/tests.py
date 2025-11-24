import numpy as np
import pytest

from src import utils
from src.classifier import AdaBoost
from src.weak_learner import DecisionStump

# --- check_data ---
def test_check_data_correct():
    X = [[0.0, 1.0], [1.0, 2.0]]
    y = [0, 1]

    X_checked, y_checked, p_checked = utils.check_data(X, y)

    assert isinstance(X_checked, np.ndarray)
    assert isinstance(y_checked, np.ndarray)
    assert X_checked.shape == (2, 2)
    assert y_checked.shape == (2,)
    assert p_checked.shape == (2,)

def test_check_data_wrong_shape():
    X = [[0.0, 1.0], [1.0, 2.0]]
    y = [0]

    with pytest.raises(ValueError):
        utils.check_data(X, y)

    X = [[[0.0, 1.0], [1.0, 2.0]]]
    y = [0, 1]

    with pytest.raises(ValueError):
        utils.check_data(X, y)


def test_check_data_wrong_empty():
    X = []
    y = []

    with pytest.raises(ValueError):
        utils.check_data(X, y)


# --- check_classes ---
def test_check_classes_correct():
    y = np.array([0, 1, 0, 1])
    utils.check_classes(y)


def test_check_classes_wrong():
    y = np.array([0, 1, 2])

    with pytest.raises(ValueError):
        utils.check_classes(y)


# --- check_norm ---
def test_check_norm_correct():
    p = np.array([0.5, 0.5])
    utils.check_norm(p)


def test_check_norm_wrong():
    with pytest.raises(ValueError):
        utils.check_norm(np.array([0.6, 0.6])) 

    with pytest.raises(ValueError):
        utils.check_norm(np.array([-0.1, 1.1]))


# --- calc_error ---
def test_calc_error():
    y = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 0, 0, 1])
    p = np.array([0.25, 0.25, 0.25, 0.25])

    # неправильно предсказан элемент 1 => 1 * 0.25 = 0.25
    eps = utils.calc_error(y, y_pred, p)
    print(eps)
    assert pytest.approx(eps) == 0.25


# --- update_weight ---
def test_update_weigh_correct_without_changes():
    y = np.array([0, 1])
    y_pred = np.array([0, 1])
    w = [0.4, 0.6]
    beta = 1.0

    p = utils.update_weight(y, y_pred, w, beta)

    assert len(p) == 2
    assert pytest.approx(sum(p)) == 1.0
    assert pytest.approx(p[0]) == 0.4
    assert pytest.approx(p[1]) == 0.6


def test_update_weigh_correct_with_changes():
    y = np.array([0, 1])
    y_pred = np.array([1, 1])
    w = [0.4, 0.6]
    beta = 0.6
    # w1 = w1 * 0.6 ^ (0 == 1) = 0.4
    # w2 = w2 * 0.6 ^ (1 == 1) = 0.6 * 0.6 = 0.36
    # после нормализации:
    # p1 = w1 / (w1 + w2) = 0.4 / (0.4 + 0.36) = 0.526
    # p2 = w2 / (w1 + w2) = 0.36 / (0.4 + 0.36) = 0.474
    # то есть из-за ошибки на первом объекте вес первого класса увеличился
    p = utils.update_weight(y, y_pred, w, beta)

    assert len(p) == 2
    assert pytest.approx(sum(p)) == 1.0
    assert pytest.approx(p[0]) == (0.4 / (0.4 + 0.36))
    assert pytest.approx(p[1]) == 0.36 / (0.4 + 0.36)

#  --- DecisionStump ---
def test_decision_stump_correct():
    # разделяем первым признаком, порог = 2
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    stump = DecisionStump()
    stump.fit(X, y)
    y_pred = stump.predict(X)

    assert stump.feature_index == 0
    assert stump.polarity == 1
    assert stump.threshold == 2
    assert y_pred.shape == y.shape
    assert np.array_equal(y_pred, y)


def test_decision_stump_wrong():
    stump = DecisionStump()
    X = np.array([[0.0], [1.0]])

    with pytest.raises(ValueError):
        stump.predict(X)


#  --- AdaBoost ---
def test_adaboost_correct():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 1, 0, 1])

    model = AdaBoost(weak_learner=DecisionStump, T=3)
    model.fit(X, y)

    stump = DecisionStump()
    stump.fit(X, y)

    assert len(model.hs) > 0
    assert len(model.betas) == len(model.hs)

    y_pred = model.predict(X)
    stump_y = stump.predict(X)

    assert set(np.unique(y_pred)) == set([0, 1])

    err_adaboost = utils.calc_error(y, y_pred)
    err_stump = utils.calc_error(y, stump_y)

    assert err_adaboost <= err_stump


