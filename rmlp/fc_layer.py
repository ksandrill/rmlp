import numpy as np


class FcLayer:
    def __init__(self, input_features_size: int, output_features_size: int, bias: bool, low: float, high: float):
        self.input_features_size = input_features_size
        self.output_features_size = output_features_size
        self.bias = bias
        input_size = input_features_size + 1 if bias else input_features_size
        self.layer_weighs = np.random.uniform(size=(output_features_size, input_size), low=low, high=high)

    def forward(self, layer_input: np.ndarray) -> (np.ndarray, np.ndarray):
        if self.bias:
            layer_input = np.hstack((layer_input, np.ones(1)))
        return self.layer_weighs @ layer_input, layer_input

    def grad(self, delta: np.ndarray, layer_input: np.ndarray, pre_activ: np.ndarray) -> (np.ndarray, np.ndarray):
        dtanh: np.ndarray = 1.0 / np.cosh(pre_activ) ** 2
        dtanh = dtanh.reshape(1, dtanh.shape[0]).T
        grad_out: np.ndarray = dtanh @ layer_input.reshape(1, layer_input.shape[0])
        grad_w: np.ndarray = (delta.reshape(1, delta.shape[0])).T * grad_out
        delta = self.layer_weighs.T @ delta
        delta = delta[:self.input_features_size]
        return grad_w, delta
