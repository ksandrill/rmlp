import pickle

import numpy as np

from draw import draw_model_real, save_param_to_html, draw_cost
from util import rmse, load_data
from three_hidden_rmlp import *

low = -0.95
high = 0.95
train_size = 6
target_size = 2

parameters_file_name = "../3hiddenLayer/model/model_2"


def unpack(array: np.ndarray):
    return np.array([i[0] for i in array]), np.array([i[1] for i in array])


def load_model(model_path: str) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float):
    with open(model_path, 'rb') as parameters_file:
        w1, w2, w3, w4, stored_lr, stored_epoch_count = pickle.load(parameters_file)
    return w1, w2, w3, w4, stored_lr, stored_epoch_count


def calc_metrics(outputs: np.ndarray, target: np.ndarray, type_str: str = "test"):
    real_g, real_kgf = unpack(target)
    model_g, model_kgf = unpack(outputs)
    rmse_g = rmse(real_g, model_g)
    rmse_kgf = rmse(real_kgf, model_kgf)
    fig_total = draw_model_real(model_g, real_g, "g_total " + type_str, "iter", "value", "red", "green")
    fig_kgf = draw_model_real(model_kgf, real_kgf, "kgf " + type_str, "iter", "value", "red", "green")
    print(type_str)
    print("g_total_rmse: ", rmse_g)
    print("kgf_rmse: ", rmse_kgf)
    save_param_to_html(fig_kgf, parameters_file_name, "kgf_" + type_str + ".html")
    save_param_to_html(fig_total, parameters_file_name, "total_" + type_str + ".html")


def unpack_size(w: np.ndarray, is_rec: bool) -> (int, int):
    layer_size = w.shape[0]
    layer_input_size = w.shape[1] - 1
    if is_rec:
        layer_input_size -= layer_size
    print(layer_input_size, layer_size)
    return layer_input_size, layer_size


def main():
    train_input, train_target, test_input, test_target = load_data("../data.csv", train_size, target_size, low, high)
    w1, w2, w3, w4, _, _ = load_model(parameters_file_name)
    input_size, r_fc1_size = unpack_size(w1, is_rec=True)
    _, r_fc2_size = unpack_size(w2, is_rec=True)
    _, r_fc3_size = unpack_size(w3, is_rec=True)
    _, output_size = unpack_size(w4, is_rec=False)
    rmlp = ThreeHiddenRmlp(input_size, r_fc1_size, r_fc2_size, r_fc3_size, output_size, 1, 0)
    rmlp.load_weights(w1, w2, w3, w4)
    res, _ = rmlp.eval(list(zip(train_input, train_target)))
    res1, test_mse = rmlp.eval(list(zip(test_input, test_target)))
    test_cost = draw_cost(np.array(test_mse), "iter", "mse", "cost_test")
    calc_metrics(np.array(res), train_target, "train")
    calc_metrics(np.array(res1), test_target, "test")
    save_param_to_html(test_cost, parameters_file_name, "test_cost.html")


if __name__ == '__main__':
    main()
