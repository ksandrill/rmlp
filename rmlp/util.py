import csv

import numpy as np


def mse(output, target) -> float:
    return ((fix_nan(output - target, 0)) ** 2).sum() / len(output)


def rmse(output, target) -> float:
    error = output - target
    error = fix_nan(error, 0.0)
    return ((error ** 2).sum() / len(output)) ** 0.5


def normalize(data: np.ndarray, low: float, high: float) -> np.ndarray:
    min_array: np.ndarray = np.nanmin(data, axis=0)
    max_array: np.ndarray = np.nanmax(data, axis=0)
    delta = max_array - min_array
    data = data - min_array
    data /= delta
    return data * (high - low) + low


def fix_nan(data: np.ndarray, value: float) -> np.ndarray:
    data[np.isnan(data)] = value
    return data


def load_data(path: str, train_size: int, target_size: int, low: float, high: float) -> (
        np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    with open(path) as fp:
        reader = csv.reader(fp, delimiter=",")
        data = np.array([[float(item) for item in line] for line in reader])
    data_input, data_target = data[:, :-target_size], data[:, -target_size:]
    data_target = normalize(data_target, low, high)
    train_input, train_target = data_input[:train_size], data_target[:train_size]
    test_input, test_target = data_input[train_size:], data_target[train_size:]

    return train_input, train_target, test_input, test_target
