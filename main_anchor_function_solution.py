from functools import partial
from typing import List, Callable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.lines import Line2D

from find_simple_baselines import find_best_coefs_for_specific_basis_functions, create_basis_functions_and_coefficients, \
    create_fourier_basis_functions_tf
from main_function_extrapolation import ExtrapolationCoefNN, reset_seeds, InfiniteExtrapolationPointMSELoss, \
    LearningRateDecay
from utils import stateless_rmse, summarize_diagnostics, reset_metrics
from main_function_extrapolation import FunctionCoefGenerator

EXTRAPOLATION_METRICS = [tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]


def configure_anchor_problem_variables(extrapolation_distance_from_approximation: float = 0):
    max_deg = 5
    generator = FunctionCoefGenerator(max_degree=max_deg,
                                      min_degree=max_deg - 1,
                                      polynomial_type="trigonometric",
                                      num_input_points=100,
                                      extrapolation_distance_from_approximation=extrapolation_distance_from_approximation)
    train_x = generator.TRAINING_POINTS.numpy()
    ext_x = generator.EXTRAPOLATION_POINTS.numpy()
    func_to_extrapolate = lambda x: 0.8 ** x - np.cos(x) + 2 * np.sin(2 * x) + 1 / (x + 1)
    train_y = func_to_extrapolate(train_x)
    ext_y = func_to_extrapolate(ext_x)
    between_x = None
    between_y = None
    if extrapolation_distance_from_approximation > 0:
        between_x = tf.convert_to_tensor(
            np.linspace(generator.training_end_point, generator.extrapolation_start_point, 100))
        between_y = func_to_extrapolate(between_x)

    return generator, train_x, train_y, ext_x, ext_y, between_x, between_y


def create_decaying_function_for_anchor_problem():
    _func_to_extrapolate = lambda x: 0.8 ** x - tf.cos(x) + 2 * tf.sin(2 * x) + 1 / (x + 1)

    func_1 = lambda x: _func_to_extrapolate(x) + 2 / (x + 1)
    func_2 = lambda x: _func_to_extrapolate(x) + (3 * tf.sin(x)) / (x + 1)
    func_3 = lambda x: _func_to_extrapolate(x) + 0.9 ** x

    return func_1, func_2, func_3


def create_non_decaying_function_for_anchor_problem():
    _func_to_extrapolate = lambda x: 0.8 ** x - tf.cos(x) + 2 * tf.sin(2 * x) + 1 / (x + 1)
    func_1 = lambda x: _func_to_extrapolate(x) + x / 10
    func_2 = lambda x: _func_to_extrapolate(x) + tf.sin(x) ** 2
    func_3 = lambda x: _func_to_extrapolate(x) + tf.math.log(x + 1) ** 2 / 5

    return func_1, func_2, func_3


def plot_decaying_anchor_functions_with_real_func(function_index: int):
    _, train_x, train_y, ext_x, ext_y, between_x, between_y = configure_anchor_problem_variables()

    func_s = create_decaying_function_for_anchor_problem()

    plt.rcParams['text.usetex'] = False
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

    func = lambda x: func_s[function_index](x)
    pred_int = func(train_x)
    pred_ext = func(ext_x)

    plt.rc('legend', fontsize=13)
    tick_size = 13.5
    plt.rc('xtick', labelsize=tick_size)
    plt.rc('ytick', labelsize=tick_size)

    _func_to_extrapolate = lambda x: 0.8 ** x - tf.cos(x) + 2 * tf.sin(2 * x) + 1 / (x + 1)
    data_x = np.linspace(0, ext_x[-1], 1000)
    plt.plot(data_x, _func_to_extrapolate(data_x), color='black', label='The Function', linewidth=2.0)
    plt.scatter(train_x, pred_int, color='orange', label='Pred Data')
    plt.scatter(ext_x, pred_ext, color='red', label='Pred Ext')

    rmse = np.round(stateless_rmse(ext_y, pred_ext), 3)

    plt.legend()
    plt.savefig(
        fr"C:\git\mainfold_approximation\results\paper_graphs\decaying_function_{function_index + 1}.png")
    plt.clf()
    a = 1


def plot_non_decaying_anchor_functions_with_real_func(function_index: int,
                                                      extrapolation_distance_from_approximation: float = 0):
    _, train_x, train_y, ext_x, ext_y, between_x, between_y = configure_anchor_problem_variables(
        extrapolation_distance_from_approximation)

    func_s = create_non_decaying_function_for_anchor_problem()

    plt.rcParams['text.usetex'] = False
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

    func = lambda x: func_s[function_index](x)
    pred_int = func(train_x)
    pred_ext = func(ext_x)

    plt.rc('legend', fontsize=13)
    tick_size = 13.5
    plt.rc('xtick', labelsize=tick_size)
    plt.rc('ytick', labelsize=tick_size)

    _func_to_extrapolate = lambda x: 0.8 ** x - tf.cos(x) + 2 * tf.sin(2 * x) + 1 / (x + 1)
    data_x = np.linspace(0, ext_x[-1], 1000)
    plt.plot(data_x, _func_to_extrapolate(data_x), color='black', label='The Function', linewidth=2.0)
    plt.scatter(train_x, pred_int, color='orange', label='Pred Data')
    plt.scatter(ext_x, pred_ext, color='red', label='Pred Ext')

    rmse = np.round(stateless_rmse(ext_y, pred_ext), 3)
    print(rmse)
    plt.legend()
    plt.savefig(
        fr"C:\git\mainfold_approximation\results\paper_graphs\non_decaying_function_{function_index + 1}.png")
    plt.clf()
    a = 1


def ls_solve_anchor_problem(basis_functions: List[Callable], extrapolation_distance_from_approximation: float = 0,
                            anchor_function_type: str = "unknown"):
    _, train_x, train_y, ext_x, ext_y, between_x, between_y = configure_anchor_problem_variables(
        extrapolation_distance_from_approximation=extrapolation_distance_from_approximation)
    coefficients = find_best_coefs_for_specific_basis_functions(train_x, train_y, basis_functions)
    print(coefficients)
    pred_func = create_basis_functions_and_coefficients(basis_functions, coefficients)
    pred_int = pred_func(train_x)
    pred_ext = pred_func(ext_x)

    _func_to_extrapolate = lambda x: 0.8 ** x - tf.cos(x) + 2 * tf.sin(2 * x) + 1 / (x + 1)
    data_x = np.linspace(0, ext_x[-1], 1000)
    plt.plot(data_x, _func_to_extrapolate(data_x), color='black', label='The Function', linewidth=2.0)

    plt.scatter(train_x, pred_int, color='orange', label='Training LS')
    plt.scatter(ext_x, pred_ext, color='red', label='Ext LS')
    plt.rc('legend', frameon=True, fontsize=13)

    # Adding labels and title to the plot
    rmse = np.round(stateless_rmse(ext_y, pred_ext), 3)
    formatted_float = '{:.3f}'.format(rmse)
    plt.legend()
    if extrapolation_distance_from_approximation > 0:
        plt.savefig(
            fr"C:\git\mainfold_approximation\results\paper_graphs\farther_away_graphs\ls_non_decaying_{extrapolation_distance_from_approximation}.png")
        plt.clf()
    else:
        plt.savefig(
            fr"C:\git\mainfold_approximation\results\paper_graphs\ls_anchor_function_{len(basis_functions)}_{anchor_function_type}.png")
        plt.clf()


def ls_solve_anchor_problem_non_decaying(num_fourier_basis_to_add: int = 7,
                                         extrapolation_distance_from_approximation: float = 0):
    func_1, func_2, func_3 = create_non_decaying_function_for_anchor_problem()
    poly_basis = [func_1, func_2, func_3] + create_fourier_basis_functions_tf(num_fourier_basis_to_add)
    ls_solve_anchor_problem(poly_basis,
                            extrapolation_distance_from_approximation=extrapolation_distance_from_approximation,
                            anchor_function_type="non_decaying")


def ls_solve_anchor_problem_decaying(num_fourier_basis_to_add: int = 7,
                                     extrapolation_distance_from_approximation: float = 0):
    func_1, func_2, func_3 = create_decaying_function_for_anchor_problem()
    poly_basis = [func_1, func_2, func_3] + create_fourier_basis_functions_tf(num_fourier_basis_to_add)
    ls_solve_anchor_problem(poly_basis,
                            extrapolation_distance_from_approximation=extrapolation_distance_from_approximation,
                            anchor_function_type="decaying")


def next_solve_anchor_problem(basis_functions: List[Callable], model_path: str,
                              extrapolation_distance_from_approximation: float = 0,
                              anchor_function_type: str = "unknown"):
    max_deg = len(basis_functions)
    _, train_x, train_y, ext_x, ext_y, between_x, between_y = configure_anchor_problem_variables(
        extrapolation_distance_from_approximation=extrapolation_distance_from_approximation)

    generator = FunctionCoefGenerator(max_degree=max_deg,
                                      min_degree=1,
                                      polynomial_type="non_basis",
                                      polynomial_basis=basis_functions,
                                      add_noise_to_input=True,
                                      create_monotonic=False,
                                      extrapolation_distance_from_approximation=extrapolation_distance_from_approximation)
    model = ExtrapolationCoefNN(num_layers=10, num_out=max_deg,
                                inner_layer_width=100,
                                generator=generator)
    dummy_input = tf.ones((1, 100))
    model(dummy_input)
    model.load_weights(model_path)
    pred_dict = model.predict(tf.reshape(train_y, (1, 100)))
    ext_pred = pred_dict['ext']
    int_pred = pred_dict['int']
    coefs = pred_dict['coefs']

    pred_func = create_basis_functions_and_coefficients(basis_functions, coefs[0])
    _func_to_extrapolate = lambda x: 0.8 ** x - tf.cos(x) + 2 * tf.sin(2 * x) + 1 / (x + 1)
    data_x = np.linspace(0, ext_x[-1], 1000)
    plt.plot(data_x, _func_to_extrapolate(data_x), color='black', label='The Function', linewidth=2.0)

    plt.scatter(train_x, int_pred[0], color='orange', label='Training NExT')
    plt.scatter(ext_x, ext_pred[0], color='red', label='Ext NExT')
    plt.rc('legend', frameon=True, fontsize=13)

    tick_size = 13.5
    plt.rc('xtick', labelsize=tick_size)
    plt.rc('ytick', labelsize=tick_size)

    print(coefs)
    rmse = round(stateless_rmse(ext_y, ext_pred[0]), 3)
    formatted_float = '{:.3f}'.format(rmse)
    plt.legend()
    if extrapolation_distance_from_approximation > 0:
        plt.savefig(
            fr"C:\git\mainfold_approximation\results\paper_graphs\farther_away_graphs\next_non_decaying_{extrapolation_distance_from_approximation}.png")
        plt.clf()
    else:
        plt.savefig(
            fr"C:\git\mainfold_approximation\results\paper_graphs\next_anchor_function_{len(basis_functions)}_{anchor_function_type}.png")
        plt.clf()
    print(rmse)


def next_solve_anchor_problem_non_decaying(num_fourier_basis_to_add: int = 7,
                                           extrapolation_distance_from_approximation: float = 0):
    model_path = f"./models/specific/non_decaying_anchor_functions_num_fillers_{num_fourier_basis_to_add}_{extrapolation_distance_from_approximation}.h5"
    func_1, func_2, func_3 = create_non_decaying_function_for_anchor_problem()
    poly_basis = [func_1, func_2, func_3] + create_fourier_basis_functions_tf(num_fourier_basis_to_add)
    next_solve_anchor_problem(poly_basis, model_path,
                              extrapolation_distance_from_approximation=extrapolation_distance_from_approximation,
                              anchor_function_type="non_decaying")


def next_solve_anchor_problem_decaying(num_fourier_basis_to_add: int = 7,
                                       extrapolation_distance_from_approximation: float = 0):
    model_path = f"./models/specific/decaying_anchor_functions_num_fillers_{num_fourier_basis_to_add}_{extrapolation_distance_from_approximation}.h5"
    func_1, func_2, func_3 = create_decaying_function_for_anchor_problem()
    poly_basis = [func_1, func_2, func_3] + create_fourier_basis_functions_tf(num_fourier_basis_to_add)
    next_solve_anchor_problem(poly_basis, model_path,
                              extrapolation_distance_from_approximation=extrapolation_distance_from_approximation,
                              anchor_function_type="decaying")


def train_next_anchor_functions_non_decaying(num_fourier_basis_to_add: int = 7,
                                             extrapolation_distance_from_approximation: float = 0):
    func_1, func_2, func_3 = create_non_decaying_function_for_anchor_problem()
    poly_basis = [func_1, func_2, func_3] + create_fourier_basis_functions_tf(num_fourier_basis_to_add)
    train_next_anchor_functions(poly_basis, f"non_decaying_anchor_functions_num_fillers_{num_fourier_basis_to_add}",
                                extrapolation_distance_from_approximation=extrapolation_distance_from_approximation)


def train_next_anchor_functions_decaying(num_fourier_basis_to_add: int = 7,
                                         extrapolation_distance_from_approximation: float = 0):
    func_1, func_2, func_3 = create_decaying_function_for_anchor_problem()
    poly_basis = [func_1, func_2, func_3] + create_fourier_basis_functions_tf(num_fourier_basis_to_add)
    train_next_anchor_functions(poly_basis, f"decaying_anchor_functions_num_fillers_{num_fourier_basis_to_add}",
                                extrapolation_distance_from_approximation=extrapolation_distance_from_approximation)


def train_next_anchor_functions(basis_functions: List[Callable], save_name: str, batch_size: int = 25,
                                norm: float = 1.0,
                                std: float = 0.25, extrapolation_distance_from_approximation: float = 0):
    max_deg = len(basis_functions)
    generator = FunctionCoefGenerator(batch_size=batch_size,
                                      max_degree=max_deg,
                                      min_degree=1,
                                      polynomial_type="non_basis",
                                      polynomial_basis=basis_functions,
                                      add_noise_to_input=False,
                                      norm_mean=norm,
                                      norm_std=std,
                                      create_monotonic=False,
                                      extrapolation_distance_from_approximation=extrapolation_distance_from_approximation)

    reset_seeds()
    model = ExtrapolationCoefNN(num_layers=10, num_out=max_deg,
                                inner_layer_width=100,
                                generator=generator)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt,
                  loss={
                      "ext": tf.keras.losses.MeanSquaredError(),
                      "coefs": tf.keras.losses.MeanSquaredError(),
                      "special": InfiniteExtrapolationPointMSELoss(
                          start_point=generator.extrapolation_start_point,
                          end_point=generator.end_point,
                          basis_len=max_deg,
                          coef_type=generator.polynomial_type,
                          basis_functions=generator.polynomial_basis)
                  },
                  metrics=reset_metrics(EXTRAPOLATION_METRICS),
                  loss_weights={"int": 0.0, "coefs": 0.0, "ext": 0.0, "special": 1.0})
    model.run_eagerly = False
    history = model.fit(generator,
                        epochs=1000,
                        batch_size=batch_size,
                        callbacks=[LearningRateDecay(decay_each_n_epochs=200, minimum_learning_rate=0.00001)],
                        verbose=2)

    summarize_diagnostics(history,
                          name=f'./results/specific/{save_name}_{extrapolation_distance_from_approximation}_1.png',
                          metrics=[metric for metric in history.history if
                                   metric in ['loss', 'special_loss', 'coefs_loss', 'ext_root_mean_squared_error',
                                              'int_root_mean_squared_error', 'coef_root_mean_squared_error']],
                          no_val=True,
                          exclude_first=0)
    summarize_diagnostics(history,
                          name=f'./results/specific/{save_name}_{extrapolation_distance_from_approximation}_2.png',
                          metrics=[metric for metric in history.history if
                                   metric in ['loss', 'special_loss', 'coefs_loss', 'ext_root_mean_squared_error',
                                              'int_root_mean_squared_error', 'coef_root_mean_squared_error']],
                          no_val=True,
                          exclude_first=200)

    model.save_weights(f"./models/specific/{save_name}_{extrapolation_distance_from_approximation}.h5")


def main():
    # plot_non_decaying_anchor_functions_with_real_func()
    # train_next_anchor_functions_decaying(0)
    # ls_solve_anchor_problem_non_decaying(0)
    # plot_non_decaying_anchor_functions_with_real_func()
    # train_next_anchor_functions_non_decaying(7, extrapolation_distance_from_approximation=7)

    plot_non_decaying_anchor_functions_with_real_func(0)
    # plot_non_decaying_anchor_functions_with_real_func(1)
    # plot_non_decaying_anchor_functions_with_real_func(2)
    #
    # plot_decaying_anchor_functions_with_real_func(0)
    # plot_decaying_anchor_functions_with_real_func(1)
    # plot_decaying_anchor_functions_with_real_func(2)
    next_solve_anchor_problem_decaying(0, extrapolation_distance_from_approximation=0)
    next_solve_anchor_problem_decaying(7, extrapolation_distance_from_approximation=0)
    # ls_solve_anchor_problem_decaying(0, extrapolation_distance_from_approximation=0)
    # ls_solve_anchor_problem_decaying(7, extrapolation_distance_from_approximation=0)

    next_solve_anchor_problem_non_decaying(0, extrapolation_distance_from_approximation=0)
    next_solve_anchor_problem_non_decaying(7, extrapolation_distance_from_approximation=0)
    # ls_solve_anchor_problem_non_decaying(0, extrapolation_distance_from_approximation=0)
    # ls_solve_anchor_problem_non_decaying(7, extrapolation_distance_from_approximation=0)

    next_solve_anchor_problem_non_decaying(7, extrapolation_distance_from_approximation=1)
    next_solve_anchor_problem_non_decaying(7, extrapolation_distance_from_approximation=3)
    next_solve_anchor_problem_non_decaying(7, extrapolation_distance_from_approximation=7)

    ls_solve_anchor_problem_non_decaying(0, extrapolation_distance_from_approximation=1)
    ls_solve_anchor_problem_non_decaying(7, extrapolation_distance_from_approximation=3)
    ls_solve_anchor_problem_non_decaying(0, extrapolation_distance_from_approximation=7)

    a = 1


if __name__ == "__main__":
    main()
