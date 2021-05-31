from draw import draw_cost, save_param_to_html
from util import load_data
import numpy as np
from three_hidden_rmlp import *

weight_low = -0.015
weight_high = 0.025
input_size = 14
r_fc1_size = 60
r_fc2_size = 45
r_fc3_size = 20
out_fc_size = 2

low = -0.95
high = 0.95
train_size = 6
target_size = 2
lr = 0.005
epoch_count = 1500
parameters_file_name = "../3hiddenLayer/model/model_3"


def main():
    train_input, train_target, test_input, test_target = load_data("../data.csv", train_size, target_size, low, high)
    rmlp = ThreeHiddenRmlp(input_size, r_fc1_size, r_fc2_size, r_fc3_size, out_fc_size, weight_low, weight_high)
    cost_list = rmlp.fit(lr, epoch_count, list(zip(train_input, train_target)), parameters_file_name)
    train_cost = draw_cost(np.array(cost_list), "epoch", "mse", "cost_train")
    save_param_to_html(train_cost, parameters_file_name, "train_cost.html")


if __name__ == '__main__':
    main()
