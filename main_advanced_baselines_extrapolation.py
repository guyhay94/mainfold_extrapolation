import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from main_function_extrapolation import FunctionCoefGenerator, reset_seeds

from typing import List, Callable
from tensorflow.python.training.tracking.data_structures import NoDependency

from utils import summarize_diagnostics, stateless_rmse, reset_metrics, average_dicts, stateless_mae
from numpy.polynomial.chebyshev import chebfit

EXTRAPOLATION_METRICS = [tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]


class EquationLearnerLayer(tf.keras.layers.Layer):
    """docstring for EquationLearnerLayer"""
    DEFAULT_FUNC_LIST = [lambda x: x, tf.sin, tf.cos, tf.sigmoid, lambda x: tf.sin(x) * tf.cos(x)]

    def __init__(self, func_list: List[Callable] = None, **kwargs):
        super().__init__(**kwargs)
        func_list = func_list if func_list is not None else EquationLearnerLayer.DEFAULT_FUNC_LIST
        self.func_list = NoDependency(func_list)

    def build(self, input_shape: tf.Tensor):
        self.units = input_shape[-1]
        self.w = self.add_weight(
            "kernel",
            shape=(self.units, self.units),
            initializer="glorot_uniform",
            trainable=True,
            regularizer=None,
        )
        self.b = self.add_weight(
            "bias",
            shape=(self.units,),
            initializer="random_normal",
            trainable=True,
            regularizer=None
        )
        self.built = True

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        self.add_loss(self.weight_regularization(self.w) + self.bias_regularization(self.b))
        x = tf.matmul(inputs, self.w) + self.b
        func_outputs = self.apply_func_list(x)
        binary_outputs = self.apply_inner_multiplication(x)
        x = tf.concat([func_outputs, binary_outputs], axis=1)
        return x

    def apply_func_list(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.concat([f(inputs) for f in self.func_list], axis=1)

    @staticmethod
    def apply_inner_multiplication(inputs: tf.Tensor) -> tf.Tensor:
        num_columns = inputs.shape[1]
        if num_columns % 2 == 0:
            even_columns = inputs[:, ::2]
            odd_columns = inputs[:, 1::2]
            return even_columns * odd_columns
        else:
            even_columns = inputs[:, :-1:2]
            odd_columns = inputs[:, 1:-1:2]
            last_column = inputs[:, -1:]
            result = even_columns * odd_columns
            return tf.concat([result, last_column], axis=1)

    def change_weight_regularization(self, l1: float = 0.0, l2: float = 0.0, regularization: Callable = None):
        self.weight_regularization = regularization if regularization is not None else tf.keras.regularizers.L1L2(l1=l1,
                                                                                                                  l2=l2)

    def change_bias_regularization(self, l1: float = 0.0, l2: float = 0.0, regularization: Callable = None):
        self.bias_regularization = regularization if regularization is not None else tf.keras.regularizers.L1L2(l1=l1,
                                                                                                                l2=l2)

    def change_both_regularization(self, l1: float = 0.0, l2: float = 0.0, regularization: Callable = None):
        self.change_weight_regularization(l1=l1, l2=l2, regularization=regularization)
        self.change_bias_regularization(l1=l1, l2=l2, regularization=regularization)

    def zero_out_low_weights(self, threshold: float = 0.01):
        to_zero_mask = np.abs(self.w) < threshold
        amount_to_zero = np.sum(to_zero_mask)
        tf.print(f"Zeroed out: {amount_to_zero} weights")
        if amount_to_zero > 0:
            self.w = tf.where(to_zero_mask, tf.zeros_like(self.w), self.w)

    def change_both_regularization_to_l0(self):
        self.change_both_regularization(regularization=lambda w: tf.reduce_sum(tf.abs(tf.sign(w))))

    def zero_out_low_weights_and_change_regularization(self, threshold: float = 0.01):
        self.zero_out_low_weights(threshold)
        self.change_both_regularization_to_l0()


def create_snake_activation(frequency=1.0):
    def snake_activation(x: tf.Tensor) -> tf.Tensor:
        return x + tf.sin(frequency * x) ** 2

    return snake_activation


class SnakeNN(tf.keras.Model):
    def __init__(self, relu: bool = False, num_out: int = 1, learnable_frequencies: bool = False):
        super().__init__()
        self.relu = relu
        self.dense_out = tf.keras.layers.Dense(num_out, activation=None)
        self.a_1 = self.add_weight(
            "frequency_2",
            shape=(1,),
            initializer="ones",
            trainable=learnable_frequencies,
            regularizer=None
        )
        self.a_2 = self.add_weight(
            "frequency_2",
            shape=(1,),
            initializer="ones",
            trainable=learnable_frequencies,
            regularizer=None
        )
        self.body = self.create_body(relu=self.relu)

    def create_body(self, relu=False) -> List[tf.keras.layers.Layer]:
        if relu:
            body = [tf.keras.layers.Dense(256, activation='relu'),
                    tf.keras.layers.Dense(64, activation='relu')]
        else:
            body = [tf.keras.layers.Dense(256, activation=create_snake_activation(self.a_1)),
                    tf.keras.layers.Dense(64, activation=create_snake_activation(self.a_2))]
            # body = [tf.keras.layers.Dense(64, activation=create_snake_activation(self.a_1)),
            #         tf.keras.layers.Dense(64, activation=create_snake_activation(self.a_2))]

        return body

    def call(self, inputs):
        x = inputs
        for layer in self.body:
            x = layer(x)
        return self.dense_out(x)


class LearningRateDecay(Callback):
    def __init__(self, decay_rate: float = 0.5, decay_each_n_epochs: int = 1000,
                 minimum_learning_rate: float = 0.00001):
        super(LearningRateDecay, self).__init__()
        self.decay_rate = decay_rate
        self.decay_each_n_epochs = decay_each_n_epochs
        self.minimum_learning_rate = minimum_learning_rate

    def on_epoch_begin(self, epoch, logs=None):
        if epoch % self.decay_each_n_epochs == 0 and epoch != 0:
            current_learning_rate = K.get_value(self.model.optimizer.lr)
            if current_learning_rate <= self.minimum_learning_rate:
                return
            new_learning_rate = current_learning_rate * self.decay_rate
            K.set_value(self.model.optimizer.lr, new_learning_rate)
            print(f"Learning rate decayed to {new_learning_rate} for epoch {epoch + 1}.")


def create_nn(learning_rate: float = 0.001, num_out: int = 1, learn_frequencies: bool = False,
              relu: bool = False):
    model = SnakeNN(num_out=num_out, relu=relu, learnable_frequencies=learn_frequencies)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()])
    model.run_eagerly = False
    return model


def benchmark_snake(train_x: np.ndarray,
                    train_y: np.ndarray,
                    ext_x: np.ndarray,
                    ext_y: np.ndarray,
                    plot: bool = False,
                    learn_frequencies: bool = False,
                    validation_set_ratio: float = 0.0):
    reset_seeds()
    model = create_nn(relu=False, learn_frequencies=learn_frequencies)
    history = model.fit(tf.reshape(train_x, (train_x.shape[0], 1)),
                        tf.reshape(train_y, (train_y.shape[0], 1)),
                        validation_split=validation_set_ratio,
                        epochs=1000,
                        batch_size=25,
                        callbacks=[LearningRateDecay(decay_each_n_epochs=200, minimum_learning_rate=0.00005),
                                   EarlyStopping(patience=50, restore_best_weights=True)])

    ext_pred = model.predict(tf.reshape(ext_x, (ext_x.shape[0], 1)))
    int_pred = model.predict(tf.reshape(train_x, (train_x.shape[0], 1)))
    if plot:
        summarize_diagnostics(history,
                              name=f'./results/specific/snake_1.png',
                              metrics=[metric for metric in history.history if
                                       metric in ['loss', 'root_mean_squared_error', 'mean_absolute_error']],
                              no_val=True,
                              exclude_first=10)
        summarize_diagnostics(history,
                              name=f'./results/specific/snake_2.png',
                              metrics=[metric for metric in history.history if
                                       metric in ['loss', 'root_mean_squared_error', 'mean_absolute_error']],
                              no_val=True,
                              exclude_first=150)

        plt.scatter(train_x, train_y, color='blue', label='Training Data')
        plt.scatter(train_x, int_pred, color='orange', label='Inter Ours Data')
        plt.scatter(ext_x, ext_y, color='green', label='Ext Data')
        plt.scatter(ext_x, ext_pred, color='red', label='Ext Data')

        # Adding labels and title to the plot
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Ext RMSE: {stateless_rmse(ext_y, ext_pred)}, int RMSE: {stateless_rmse(train_y, int_pred)}')
        plt.legend()
        plt.show()

    return int_pred, ext_pred, model


def benchmark_relu(train_x: np.ndarray,
                   train_y: np.ndarray,
                   ext_x: np.ndarray,
                   ext_y: np.ndarray,
                   plot: bool = False):
    reset_seeds()
    model = create_nn(relu=True)
    history = model.fit(tf.reshape(train_x, (train_x.shape[0], 1)),
                        tf.reshape(train_y, (train_y.shape[0], 1)),
                        epochs=1000,
                        batch_size=125,
                        callbacks=[LearningRateDecay(decay_each_n_epochs=200, minimum_learning_rate=0.00005)])

    ext_pred = model.predict(tf.reshape(ext_x, (ext_x.shape[0], 1)))
    int_pred = model.predict(tf.reshape(train_x, (train_x.shape[0], 1)))
    if plot:
        summarize_diagnostics(history,
                              name=f'./results/specific/snake_1.png',
                              metrics=[metric for metric in history.history if
                                       metric in ['loss', 'root_mean_squared_error', 'mean_absolute_error']],
                              no_val=True,
                              exclude_first=10)
        summarize_diagnostics(history,
                              name=f'./results/specific/snake_2.png',
                              metrics=[metric for metric in history.history if
                                       metric in ['loss', 'root_mean_squared_error', 'mean_absolute_error']],
                              no_val=True,
                              exclude_first=150)

        plt.scatter(train_x, train_y, color='blue', label='Training Data')
        plt.scatter(train_x, int_pred, color='orange', label='Inter Ours Data')
        plt.scatter(ext_x, ext_y, color='green', label='Ext Data')
        plt.scatter(ext_x, ext_pred, color='red', label='Ext Data')

        # Adding labels and title to the plot
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Ext RMSE: {stateless_rmse(ext_y, ext_pred)}, int RMSE: {stateless_rmse(train_y, int_pred)}')
        plt.legend()
        plt.show()

    return int_pred, ext_pred


def benchmark_relu_network_on_anchor_problem():
    generator = FunctionCoefGenerator(max_degree=7,
                                      min_degree=6 - 1,
                                      polynomial_type="trigonometric",
                                      num_input_points=100)
    ext_x = np.linspace(generator.extrapolation_start_point, generator.end_point, 100)
    train_x = generator.TRAINING_POINTS.numpy()
    _func_to_extrapolate = lambda x: 0.8 ** x - tf.cos(x) + 2 * tf.sin(2 * x) + 1 / (x + 1)
    train_y = _func_to_extrapolate(train_x)
    ext_y = _func_to_extrapolate(ext_x)

    pred_int, pred_ext = benchmark_relu(train_x, train_y, ext_x, ext_y)
    data_x = np.linspace(0, ext_x[-1], 1000)
    plt.plot(data_x, _func_to_extrapolate(data_x), color='black', label='The Function', linewidth=2.0)
    plt.scatter(train_x, pred_int, color='orange', label='Training ReLU')
    plt.scatter(ext_x, pred_ext, color='red', label='Ext ReLU')

    rmse = np.round(stateless_rmse(ext_y, pred_ext), 3)
    print(rmse)
    # plt.xlabel('X')
    # plt.ylabel('Y')
    plt.title(f'ReLU, Ext RMSE: {str(rmse)}')
    plt.rc('legend', fontsize=13)
    tick_size = 13.5
    plt.rc('xtick', labelsize=tick_size)
    plt.rc('ytick', labelsize=tick_size)
    plt.legend()
    plt.savefig(
        fr"C:\git\mainfold_approximation\results\paper_graphs\relu_anchor_function.png")
    plt.clf()


def benchmark_snake_network_on_anchor_problem(extrapolation_distance_from_approximation: float = 0,
                                              learn_frequencies: bool = True,
                                              validation_set_ratio: float = 0.0):
    generator = FunctionCoefGenerator(max_degree=7,
                                      min_degree=6 - 1,
                                      polynomial_type="trigonometric",
                                      num_input_points=100,
                                      extrapolation_distance_from_approximation=extrapolation_distance_from_approximation)
    ext_x = np.linspace(generator.extrapolation_start_point, generator.end_point, 100)
    train_x = generator.TRAINING_POINTS.numpy()
    _func_to_extrapolate = lambda x: 0.8 ** x - tf.cos(x) + 2 * tf.sin(2 * x) + 1 / (x + 1)
    train_y = _func_to_extrapolate(train_x)
    ext_y = _func_to_extrapolate(ext_x)

    pred_int, pred_ext, model = benchmark_snake(train_x, train_y, ext_x, ext_y,
                                                plot=False, learn_frequencies=learn_frequencies,
                                                validation_set_ratio=validation_set_ratio)

    data_x = np.linspace(0, ext_x[-1], 1000)
    plt.plot(data_x, _func_to_extrapolate(data_x), color='black', label='The Function', linewidth=2.0)
    plt.scatter(train_x, pred_int, color='orange', label='Training Snake')
    plt.scatter(ext_x, pred_ext, color='red', label='Ext Snake')

    rmse = np.round(stateless_rmse(ext_y, pred_ext), 3)
    print(rmse)

    plt.rc('legend', frameon=True, fontsize=13)
    tick_size = 13.5
    plt.rc('xtick', labelsize=tick_size)
    plt.rc('ytick', labelsize=tick_size)
    plt.legend()
    if extrapolation_distance_from_approximation > 0:
        plt.savefig(
            fr"C:\git\mainfold_approximation\results\paper_graphs\farther_away_graphs\snake_non_decaying_{extrapolation_distance_from_approximation}.png")
        plt.clf()
    else:
        plt.savefig(
            fr"C:\git\mainfold_approximation\results\paper_graphs\snake_anchor_function.png")
        plt.clf()
    a = 1


def benchmark_snake_network_on_noisy_chebyshev_problem(deg: int):
    generator = FunctionCoefGenerator(max_degree=deg,
                                      min_degree=deg - 1,
                                      polynomial_type="chebyshev",
                                      num_input_points=100)
    ext_x = np.linspace(generator.extrapolation_start_point, generator.end_point, 100)
    metrics = [stateless_rmse, stateless_mae]
    max_rmse = 0
    results = {}
    results_list = []
    df = pd.read_pickle(
        r"C:\git\mainfold_approximation\dataset\coef_extrapolation\chebyshev\test_data_noise_" + f"{deg}_snr_35_norm_regular.pickle")
    for i in range(df.shape[0]):
        print(i)
        int_train = df['int_train'][i]
        int_data = df['int_clean'][i]
        ext_data = df['ext'][i]
        train_x = generator.TRAINING_POINTS.numpy()
        train_y = int_train

        pred_int, pred_ext, model = benchmark_snake(train_x, train_y[0], ext_x, ext_data[0], plot=False,
                                                    learn_frequencies=True)
        pred_ext = tf.expand_dims(tf.squeeze(pred_ext), axis=0)
        pred_int = tf.expand_dims(tf.squeeze(pred_int), axis=0)
        res_old = {f"ext_full_{metric.__name__}": metric(ext_data, pred_ext) for metric in
                   metrics}
        res_old['int_root_mean_squared_error_clean'] = stateless_rmse(int_data, pred_int)
        current_rmse = stateless_rmse(pred_ext, ext_data)
        if max_rmse < current_rmse:
            max_rmse = current_rmse

        results_list.append(res_old)

    average_results = average_dicts(results_list)
    results[f"deg_7_int_0_is_coef_0_ext_0_special_1"] = average_results
    df_results = pd.DataFrame(results).transpose()
    df_results.to_csv(f"./results/chebyshev/chebyshev_only_max_deg_{deg}_y_prediction_snake_learnable.csv")
    print("Done")
    a = 1


def benchmark_relu_network_on_noisy_chebyshev_problem(deg: int):
    generator = FunctionCoefGenerator(max_degree=deg,
                                      min_degree=deg - 1,
                                      polynomial_type="chebyshev",
                                      num_input_points=100)
    ext_x = np.linspace(generator.extrapolation_start_point, generator.end_point, 100)
    metrics = [stateless_rmse, stateless_mae]
    max_rmse = 0
    results = {}
    results_list = []
    df = pd.read_pickle(
        r"C:\git\mainfold_approximation\dataset\coef_extrapolation\chebyshev\test_data_noise_" + f"{deg}_snr_35_norm_regular.pickle")
    for i in range(df.shape[0]):
        print(i)
        int_train = df['int_train'][i]
        int_data = df['int_clean'][i]
        ext_data = df['ext'][i]
        train_x = generator.TRAINING_POINTS.numpy()
        train_y = int_train

        pred_int, pred_ext = benchmark_relu(train_x, train_y[0], ext_x, ext_data[0], plot=False)
        pred_ext = tf.expand_dims(tf.squeeze(pred_ext), axis=0)
        pred_int = tf.expand_dims(tf.squeeze(pred_int), axis=0)
        res_old = {f"ext_full_{metric.__name__}": metric(ext_data, pred_ext) for metric in
                   metrics}
        res_old['int_root_mean_squared_error_clean'] = stateless_rmse(int_data, pred_int)
        current_rmse = stateless_rmse(pred_ext, ext_data)
        if max_rmse < current_rmse:
            max_rmse = current_rmse

        results_list.append(res_old)

    average_results = average_dicts(results_list)
    results[f"deg_7_int_0_is_coef_0_ext_0_special_1"] = average_results
    df_results = pd.DataFrame(results).transpose()
    df_results.to_csv(f"./results/chebyshev/chebyshev_only_max_deg_{deg}_y_prediction_relu.csv")
    print("Done")


def benchmark_relu_network_on_noisy_chebyshev_problem_different_noises():
    max_deg = 5
    generator = FunctionCoefGenerator(max_degree=max_deg,
                                      min_degree=max_deg - 1,
                                      polynomial_type="chebyshev",
                                      num_input_points=100)
    ext_x = np.linspace(generator.extrapolation_start_point, generator.end_point, 100)
    metrics = [stateless_rmse, stateless_mae]
    max_rmse = 0
    results = {}
    results_list = []
    for noise_level in ["no_noise", 50, 40, 35, 30, 20]:
        df = pd.read_pickle(
            r"C:\git\mainfold_approximation\dataset\coef_extrapolation\chebyshev\test_data_noise_" + f"{max_deg}_snr_{noise_level}_norm_regular.pickle")
        for i in range(df.shape[0]):
            print(i)
            int_train = df['int_train'][i]
            int_data = df['int_clean'][i]
            ext_data = df['ext'][i]
            train_x = generator.TRAINING_POINTS.numpy()
            train_y = int_train

            pred_int, pred_ext = benchmark_relu(train_x, train_y[0], ext_x, ext_data[0], plot=False)
            pred_ext = tf.expand_dims(tf.squeeze(pred_ext), axis=0)
            pred_int = tf.expand_dims(tf.squeeze(pred_int), axis=0)
            res_old = {f"ext_full_{metric.__name__}": metric(ext_data, pred_ext) for metric in
                       metrics}
            res_old['int_root_mean_squared_error_clean'] = stateless_rmse(int_data, pred_int)
            current_rmse = stateless_rmse(pred_ext, ext_data)
            if max_rmse < current_rmse:
                max_rmse = current_rmse

            results_list.append(res_old)

        average_results = average_dicts(results_list)
        results[f"relu_on_deg_{max_deg}_noise_level_{noise_level}"] = average_results
    df_results = pd.DataFrame(results).transpose()
    df_results.to_csv(f"./results/chebyshev/chebyshev_only_max_deg_{max_deg}_y_prediction_relu.csv")
    print("Done")


def benchmark_snake_network_on_noisy_chebyshev_problem_different_noises():
    max_deg = 5
    generator = FunctionCoefGenerator(max_degree=max_deg,
                                      min_degree=max_deg - 1,
                                      polynomial_type="chebyshev",
                                      num_input_points=100)
    ext_x = np.linspace(generator.extrapolation_start_point, generator.end_point, 100)
    metrics = [stateless_rmse, stateless_mae]
    max_rmse = 0
    results = {}
    results_list = []
    for noise_level in ["no_noise", 50, 40, 35, 30, 20]:
        df = pd.read_pickle(
            r"C:\git\mainfold_approximation\dataset\coef_extrapolation\chebyshev\test_data_noise_" + f"{max_deg}_snr_{noise_level}_norm_regular.pickle")
        for i in range(df.shape[0]):
            print(i)
            int_train = df['int_train'][i]
            int_data = df['int_clean'][i]
            ext_data = df['ext'][i]
            train_x = generator.TRAINING_POINTS.numpy()
            train_y = int_train

            pred_int, pred_ext, model = benchmark_snake(train_x, train_y[0], ext_x, ext_data[0], plot=False)
            pred_ext = tf.expand_dims(tf.squeeze(pred_ext), axis=0)
            pred_int = tf.expand_dims(tf.squeeze(pred_int), axis=0)
            res_old = {f"ext_full_{metric.__name__}": metric(ext_data, pred_ext) for metric in
                       metrics}
            res_old['int_root_mean_squared_error_clean'] = stateless_rmse(int_data, pred_int)
            current_rmse = stateless_rmse(pred_ext, ext_data)
            if max_rmse < current_rmse:
                max_rmse = current_rmse

            results_list.append(res_old)

        average_results = average_dicts(results_list)
        results[f"snake_on_deg_{max_deg}_noise_level_{noise_level}"] = average_results
    df_results = pd.DataFrame(results).transpose()
    df_results.to_csv(f"./results/chebyshev/chebyshev_only_max_deg_{max_deg}_y_prediction_snake.csv")
    print("Done")


def main():
    benchmark_snake_network_on_noisy_chebyshev_problem(5)
    benchmark_relu_network_on_noisy_chebyshev_problem(5)
    benchmark_relu_network_on_anchor_problem()
    benchmark_snake_network_on_anchor_problem(learn_frequencies=True,
                                              extrapolation_distance_from_approximation=1,
                                              validation_set_ratio=0.0)
    benchmark_snake_network_on_anchor_problem(learn_frequencies=True,
                                              extrapolation_distance_from_approximation=3,
                                              validation_set_ratio=0.0)
    benchmark_snake_network_on_anchor_problem(learn_frequencies=True,
                                              extrapolation_distance_from_approximation=7,
                                              validation_set_ratio=0.0)
    benchmark_snake_network_on_anchor_problem(learn_frequencies=True,
                                              extrapolation_distance_from_approximation=0,
                                              validation_set_ratio=0.0)

    benchmark_relu_network_on_anchor_problem()


if __name__ == "__main__":
    main()
