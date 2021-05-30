from draw import *
from rmlp import Rmlp
from util import *

weight_low = -0.01
weight_high = 0.03
input_size = 14
r_fc1_size = 42
r_fc2_size = 69
out_fc_size = 2

low = -1
high = 1
train_size = 6
target_size = 2
lr = 0.01
epoch_count = 1500


def unpack(array: np.ndarray):
    return np.array([i[0] for i in array]), np.array([i[1] for i in array])


def calc_metrics(outputs: np.ndarray, target: np.ndarray, type_str: str = "test"):
    real_g, real_kgf = unpack(target)
    model_g, model_kgf = unpack(outputs)
    rmse_g = rmse(real_g, model_g)
    rmse_kgf = rmse(real_kgf, model_kgf)
    draw_model_real(model_g, real_g, "g_total " + type_str, "iter", "value", "red", "green")
    draw_model_real(model_kgf, real_kgf, "kgf " + type_str, "iter", "value", "red", "green")
    print(type_str)
    print("g_total_rmse: ", rmse_g)
    print("kgf_rmse: ", rmse_kgf)


def main():
    train_input, train_target, test_input, test_target = load_data("data.csv", train_size, target_size)
    train_target = normalize(train_target, low, high)
    test_target = normalize(test_target, low, high)
    rmlp = Rmlp(input_size, r_fc1_size, r_fc2_size, out_fc_size, weight_low, weight_high)
    cost_list = rmlp.fit(lr, epoch_count, list(zip(train_input, train_target)))
    draw_cost(np.array(cost_list), "epoch", "mse", "cost")
    res = rmlp.eval(list(zip(train_input, train_target)))
    res1 = rmlp.eval(list(zip(test_input, test_target)))
    calc_metrics(np.array(res), train_target, "train")
    calc_metrics(np.array(res1), test_target, "test")

    # pyplot.plot(train_input, res)
    # pyplot.plot(train_input, train_target)
    # pyplot.show()
    # pyplot.clf()
    # pyplot.plot(test_input, res1)
    # pyplot.plot(test_input, test_target)
    # pyplot.show()


if __name__ == '__main__':
    main()
