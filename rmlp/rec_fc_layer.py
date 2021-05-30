import numpy as np


class RecFcLayer:
    def __init__(self, input_features_size: int, output_features_size: int, bias: bool, low: float, high: float):
        self.input_features_size = input_features_size
        self.output_features_size = output_features_size
        self.bias = bias
        input_size = input_features_size + 1 if bias else input_features_size
        self.layer_weighs = np.random.uniform(size=(output_features_size, output_features_size + input_size), low=low,
                                              high=high)

    def forward(self, layer_input: np.ndarray, h: np.ndarray) -> (np.ndarray, np.ndarray):
        if self.bias:
            layer_input = np.hstack((layer_input, np.ones(1)))
        layer_input = np.hstack((layer_input, h))
        return self.layer_weighs @ layer_input, layer_input

    def grad(self, error: np.ndarray, layer_input: np.ndarray, prev_grad_out: np.ndarray,
             pre_activ_out: np.ndarray) -> (
            np.ndarray, np.ndarray, np.ndarray):
        output_size: int = self.output_features_size
        input_size: int = self.input_features_size
        dtanh: np.ndarray = 1.0 / np.cosh(pre_activ_out) ** 2
        dtanh = dtanh.reshape(1, dtanh.shape[0]).T
        grad_out: np.ndarray = dtanh @ layer_input.reshape(1, layer_input.shape[0])
        grad_out[:, input_size:input_size + output_size] += prev_grad_out[:, input_size:input_size + output_size] * self.layer_weighs[:, input_size:input_size + output_size] * dtanh # хвостик
        grad_w: np.ndarray = (error.reshape(1, error.shape[0])).T * grad_out
        error = self.layer_weighs.T @ error
        error = error[:input_size]
        return grad_w, grad_out, error
