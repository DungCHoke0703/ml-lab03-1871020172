"""Các hàm nhỏ giúp hiểu cơ chế Gaussian Naive Bayes.

Sinh viên chỉ hoàn thiện hai hàm dưới đây; không cần tự cài toàn bộ GaussianNB.
"""

import math


def compute_class_priors(y):
    """Trả về dict {class_label: prior_probability}."""
    total = len(y)
    priors = {}
    for label in y:
        priors[label] = priors.get(label, 0) + 1
    for label in priors:
        priors[label] = priors[label] / total
    return priors


def gaussian_log_likelihood(x, mean, var):
    """Tính log p(x | class) với giả định Gaussian một chiều.

    Công thức:
        -0.5 * log(2*pi*var) - (x-mean)^2 / (2*var)
    """
    if var <= 0:
        raise ValueError(f"Variance phải dương, nhận được var={var}")
    return -0.5 * math.log(2 * math.pi * var) - ((x - mean) ** 2) / (2 * var)
