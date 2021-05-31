import pickle

import numpy as np

from fc_layer import FcLayer
from rec_fc_layer import RecFcLayer
from util import fix_nan
from util import mse

Outputs: type = list[np.ndarray, np.ndarray, np.ndarray]
OutputsWithoutActiv: type = list[np.ndarray, np.ndarray, np.ndarray]
Inputs: type = list[np.ndarray, np.ndarray, np.ndarray]
DataSet: type = list[(np.ndarray, np.ndarray)]
Gradients: type = list[np.ndarray, np.ndarray, np.ndarray]
LoopOutGrads: type = list[np.ndarray, np.ndarray]


class Rmlp:
    def __init__(self, input_size: int, first_layer_size: int, second_layer_size: int, output_layer_size: int,
                 low: float, high: float):
        self.r_fc1 = RecFcLayer(input_size, first_layer_size, bias=True, low=low, high=high)
        self.r_fc2 = RecFcLayer(first_layer_size, second_layer_size, bias=True, low=low, high=high)
        self.output_fc = FcLayer(second_layer_size, output_layer_size, bias=True, low=low, high=high)

    def forward(self, input_data: np.ndarray, prev_outputs: Outputs) -> (Inputs, OutputsWithoutActiv, Outputs):
        h1, h2, _ = prev_outputs
        pre_out1, input1 = self.r_fc1.forward(input_data, h1)
        h1 = np.tanh(pre_out1)
        pre_out2, input2 = self.r_fc2.forward(h1, h2)
        h2 = np.tanh(pre_out2)
        pre_out3, input3 = self.output_fc.forward(h2)
        h3 = np.tanh(pre_out3)
        return [input1, input2, input3], [pre_out1, pre_out2, pre_out3], [h1, h2, h3]

    def _update_weights(self, gradients: Gradients, lr: float):
        self.r_fc1.layer_weighs -= lr * gradients[0]
        self.r_fc2.layer_weighs -= lr * gradients[1]
        self.output_fc.layer_weighs -= lr * gradients[2]

    def load_weights(self, w1: np.ndarray, w2: np.ndarray, w3: np.ndarray):
        self.r_fc1.layer_weighs = w1
        self.r_fc2.layer_weighs = w2
        self.output_fc.layer_weighs = w3

    def save_data(self, parameters_file_name, lr, epoch_count=0):
        with open(parameters_file_name, 'wb') as parameters_file:
            w1 = self.r_fc1.layer_weighs
            w2 = self.r_fc2.layer_weighs
            w3 = self.output_fc.layer_weighs
            pickle.dump([w1, w2, w3, lr, epoch_count], parameters_file)

    def fit(self, lr, epoch_count, train_set: DataSet, parameters_file_name: str) -> list[float]:
        mse_list = []
        self.save_data(parameters_file_name + "_start", lr)
        for i in range(epoch_count):
            print("epoch: ", i + 1, end=' ')
            prev_outputs: Outputs = [np.zeros(self.r_fc1.output_features_size),
                                     np.zeros(self.r_fc2.output_features_size),
                                     np.zeros(self.output_fc.output_features_size)]
            prev_grad_out2 = np.zeros_like(self.r_fc2.layer_weighs)
            prev_grad_out1 = np.zeros_like(self.r_fc1.layer_weighs)
            grads: Gradients = [np.zeros_like(self.r_fc1.layer_weighs), np.zeros_like(self.r_fc2.layer_weighs),
                                np.zeros_like(self.output_fc.layer_weighs)]
            avg_mse = 0.0
            for item in train_set:
                x, y = item
                [input1, input2, input3], [v1, v2, v3], prev_outputs = self.forward(x, prev_outputs)
                error = prev_outputs[-1] - y
                w3_grad, error = self.output_fc.grad(error, input3, v3)
                w2_grad, prev_grad_out2, error = self.r_fc2.grad(error, input2, prev_grad_out2, v2)
                w1_grad, prev_grad_out1, error = self.r_fc1.grad(error, input1, prev_grad_out1, v1)
                grads[0] += w1_grad
                grads[1] += w2_grad
                grads[2] += w3_grad
                avg_mse += mse(prev_outputs[-1], y)
            avg_mse /= len(train_set)
            print(avg_mse)
            mse_list.append(avg_mse)
            self._update_weights(grads, lr)
        self.save_data(parameters_file_name, lr, epoch_count)
        return mse_list

    def eval(self, test_data: DataSet) -> (list[np.ndarray], list[float]):
        prev_outputs: Outputs = [np.zeros(self.r_fc1.output_features_size),
                                 np.zeros(self.r_fc2.output_features_size),
                                 np.zeros(self.output_fc.output_features_size)]
        res = []
        error_list = []
        for item in test_data:
            x, y = item
            x = fix_nan(x, 0.0)
            _, _, prev_outputs = self.forward(x, prev_outputs)
            res.append(prev_outputs[-1])
            error_list.append(mse(prev_outputs[-1], y))
        return res, error_list
