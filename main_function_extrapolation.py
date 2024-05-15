import os
import random as rn
import pandas as pd
import tensorflow as tf
import numpy as np
from scipy.integrate import quad, dblquad

from matplotlib import pyplot as plt
from typing import Tuple, Callable, List, Dict

from extrapolation_poly_layer import PolynomialCoefLayer
from find_simple_baselines import create_fourier_basis_functions_tf, create_chebyshev_basis_functions
from utils import summarize_diagnostics, stateless_rmse, reset_metrics, stateless_mae, average_dicts

from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
from numpy.polynomial.chebyshev import chebfit
import gc
from keras import backend as K

EXTRAPOLATION_METRICS = [tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]


def monotonic_increasing_loss(y_true, y_pred, rtol: float = 10 ** -3):
    """
    Custom loss function that checks if y_pred is monotonic increasing over the points.

    Arguments:
    y_true -- The true values of y for the given set of points (ignored).
    y_pred -- The predicted values of y for the given set of points.
    """
    diff = tf.nn.relu(y_pred[:-1] - y_pred[1:] - rtol)
    loss_value = tf.reduce_sum(diff)
    return loss_value


class MultiActivationDense(tf.keras.layers.Layer):
    def __init__(self, units, activations, **kwargs):
        super(MultiActivationDense, self).__init__(**kwargs)
        self.units = units
        self.activations = activations
        self.len_activations = len(activations)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[-1], self.units),
                                      initializer='glorot_uniform',
                                      trainable=True)
        super(MultiActivationDense, self).build(input_shape)

    def call(self, inputs):
        output = tf.keras.backend.dot(inputs, self.kernel)
        output_multi = tf.stack(
            [self.activations[i % self.len_activations](output[:, i]) for i in range(output.shape[-1])], axis=1)
        return output_multi

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)


class FunctionGenerator(tf.keras.utils.Sequence):
    def __init__(self, batch_size, max_degree: int = 25, num_out: int = 30):
        self.batch_size = batch_size
        self.dimension = 1
        self.training_data = tf.convert_to_tensor(np.linspace(0, 0.7, 100))
        self.extrapolation_data = tf.convert_to_tensor(np.linspace(0.7, 1, num_out))
        self.max_degree = max_degree

    def __len__(self) -> int:
        return 100

    def create_batch_coefs(self) -> np.ndarray:
        # frequencies = np.random.rand(self.batch_size, 1) * self.frequency_range
        degrees = np.random.randint(low=1, high=self.max_degree + 1, size=(self.batch_size, 1))
        return np.apply_along_axis(self.generate_random_sobolove_functions, 1, degrees)

    @staticmethod
    def generate_sin_function(frequency: float) -> Callable:
        return lambda x: tf.sin(frequency * x)

    @staticmethod
    def generate_cos_function(frequency: float) -> Callable:
        return lambda x: tf.cos(frequency * x)

    def generate_random_sin_cos_with_frequency(self, frequency: float) -> Callable:
        if np.random.rand() < 0.5:
            return self.generate_sin_function(frequency)
        else:
            return self.generate_cos_function(frequency)

    @staticmethod
    def generate_random_sobolove_functions(degree: int):
        coefs = np.random.randn(int(degree), 1)
        normalized_coefs = coefs / np.linalg.norm(coefs, ord=2)

        point = float(np.random.uniform(0, 0.7))

        def polynomial(x):
            return np.polyval(normalized_coefs, x - point)

        return polynomial

    def create_val_set(self, frequency: float, sin: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
        func = self.generate_sin_function(frequency) if sin else self.generate_cos_function(frequency)
        return self.create_val_set_from_func(func)

    def create_val_set_from_func(self, func: Callable) -> Tuple[tf.Tensor, tf.Tensor]:
        val_x = tf.expand_dims(func(self.training_data), 0)
        val_y = tf.expand_dims(func(self.extrapolation_data), 0)
        return val_x, val_y

    def __getitem__(self, index) -> Tuple[tf.Tensor, tf.Tensor]:
        funcs = self.create_batch_coefs()
        train_x = tf.stack([f(self.training_data) for f in funcs])
        train_y = tf.stack([f(self.extrapolation_data) for f in funcs])
        return train_x, train_y


class FunctionCoefGenerator(tf.keras.utils.Sequence):

    def __init__(self, batch_size: int = 25, max_degree: int = 25, polynomial_type: str = "chebyshev",
                 create_monotonic: bool = False, coef_only: bool = False, snr_level: int = 35,
                 norm_only: bool = False, norm_mean: float = 1.0, norm_std: float = 0.25,
                 min_degree: int = 1, polynomial_basis: List[Callable] = None,
                 add_noise_to_input: bool = True, num_input_points: int = 100,
                 extrapolation_distance_from_approximation: float = 0):
        self.batch_size = batch_size
        self.dimension = 1
        self.max_degree = max_degree
        self.min_degree = min_degree
        self.polynomial_type = polynomial_type
        self.create_monotonic = create_monotonic

        self.polynomial_basis = polynomial_basis
        self.poly_layer = PolynomialCoefLayer(polynomial_type=polynomial_type,
                                              polynomial_basis=polynomial_basis)
        self.coef_only = coef_only
        self.norm_only = norm_only
        self.snr_level = snr_level
        self.start_point = None
        self.training_end_point = None
        self.extrapolation_start_point = None
        self.end_point = None
        self.second_dimension_ext_start_point = None
        self.second_dimension_ext_end_point = None
        self.TRAINING_POINTS, self.EXTRAPOLATION_POINTS, self.FULL_POINTS = self.get_points(self.polynomial_type,
                                                                                            extrapolation_distance_from_approximation,
                                                                                            num_input_points)

        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.add_noise_to_input = add_noise_to_input

    def __len__(self) -> int:
        return 100

    @staticmethod
    def get_start_end_points(polynomial_type: str, extrapolation_distance_from_approximation: float = 0):
        if polynomial_type in ["trigonometric", "non_basis"]:
            start = 0
            end_train = 1.5 * np.pi
            start_ext = 1.5 * np.pi
            end_ext = 2 * np.pi
        elif polynomial_type == "spherical_harmonics":
            # theta is longitudinal
            # phi is colatitudinal
            start_theta_train = 0
            end_theta_train = 2 * np.pi
            start_phi_train = np.pi
            end_phi_train = 2 * np.pi / 3
            start_theta_ext = start_theta_train
            end_theta_ext = end_theta_train
            start_phi_ext = end_phi_train - extrapolation_distance_from_approximation
            end_phi_ext = 0
            return start_theta_train, end_theta_train, start_phi_train, end_phi_train, start_theta_ext, end_theta_ext, start_phi_ext, end_phi_ext
        else:
            start = -1
            end_train = 0.5
            start_ext = 0.5
            end_ext = 1
        return start, end_train, start_ext + extrapolation_distance_from_approximation, end_ext + extrapolation_distance_from_approximation

    def get_points(self, polynomial_type: str, extrapolation_distance_from_approximation: float = 0,
                   num_input_points: int = 100):
        if polynomial_type == "spherical_harmonics":
            start_theta_train, end_theta_train, start_phi_train, end_phi_train, start_theta_ext, end_theta_ext, start_phi_ext, end_phi_ext = self.get_start_end_points(
                polynomial_type, extrapolation_distance_from_approximation)
            theta_train_points = np.linspace(start_theta_train, end_theta_train, int(num_input_points ** 0.5))
            phi_train_points = np.linspace(start_phi_train, end_phi_train, int(num_input_points ** 0.5))
            theta_train_points_total, phi_train_points_total = np.meshgrid(theta_train_points, phi_train_points)
            self.TRAINING_POINTS = tf.concat(
                [theta_train_points_total.reshape((-1, 1)), phi_train_points_total.reshape((-1, 1))], axis=1)

            theta_ext_points = np.linspace(start_theta_ext, end_theta_ext, 100)
            phi_ext_points = np.linspace(start_phi_ext, end_phi_ext, 100)
            theta_ext_points_total, phi_ext_points_total = np.meshgrid(theta_ext_points, phi_ext_points)
            self.EXTRAPOLATION_POINTS = tf.concat(
                [theta_ext_points_total.reshape((-1, 1)), phi_ext_points_total.reshape((-1, 1))], axis=1)
            self.FULL_POINTS = None  # used only for monotonic chebyshev
            self.extrapolation_start_point = start_theta_ext
            self.end_point = end_theta_ext
            self.second_dimension_ext_start_point = start_phi_ext
            self.second_dimension_ext_end_point = end_phi_ext
        else:
            self.start_point, self.training_end_point, self.extrapolation_start_point, self.end_point = self.get_start_end_points(
                polynomial_type, extrapolation_distance_from_approximation)
            self.EXTRAPOLATION_POINTS = tf.convert_to_tensor(
                np.linspace(self.extrapolation_start_point, self.end_point, 100))
            self.TRAINING_POINTS = tf.convert_to_tensor(
                np.linspace(self.start_point, self.training_end_point, num_input_points))
            self.FULL_POINTS = tf.concat([self.TRAINING_POINTS, self.EXTRAPOLATION_POINTS], axis=0)

        return self.TRAINING_POINTS, self.EXTRAPOLATION_POINTS, self.FULL_POINTS

    def create_batch_coefs(self) -> tf.Tensor:
        if self.create_monotonic:
            if self.polynomial_type != "chebyshev":
                raise NotImplementedError(
                    f"{self.polynomial_type} not an implemented polynomial for monotonic generation")
            return self.create_chebyshev_monotonic_batch_coefs()
        else:
            return self.create_basic_batch_coefs_fast()

    def create_chebyshev_monotonic_batch_coefs(self) -> tf.Tensor:
        coefs = self.create_basic_batch_coefs_fast(max_degree=self.max_degree - 1)
        if tf.shape(coefs)[1] < self.max_degree:
            pad_width = self.max_degree - tf.shape(coefs)[1]
            coefs = tf.pad(coefs, [[0, 0], [0, pad_width]])
        full_func_outputs = self.poly_layer.call([coefs, tf.cast(self.FULL_POINTS, coefs.dtype)])
        func_mins = tf.reduce_min(full_func_outputs, axis=1)
        correcting_terms = tf.where(func_mins < 0, tf.abs(func_mins) + 0.0001, tf.zeros_like(func_mins))
        corrected_coefs = coefs + tf.concat([tf.expand_dims(correcting_terms, axis=1), tf.zeros_like(coefs[:, 1:])],
                                            axis=1)
        monotonic_coefs = tf.map_fn(self.integrate_chebyshev_coefs, corrected_coefs)
        return monotonic_coefs

    def integrate_chebyshev_coefs(self, coefs: tf.Tensor) -> tf.Tensor:
        integrated_coefs = tf.zeros_like(coefs)
        for i in range(coefs.shape[0]):
            if i == 0:
                integrated_coefs = tf.tensor_scatter_nd_add(integrated_coefs, [[1]], [coefs[0]])
            elif i == 1:
                integrated_coefs = tf.tensor_scatter_nd_add(integrated_coefs, [[0], [2]],
                                                            [0.25 * coefs[1], 0.25 * coefs[1]])
            else:
                if coefs[i] == 0:
                    continue
                integrated_coefs = tf.tensor_scatter_nd_add(integrated_coefs, [[i + 1]], [0.5 * coefs[i] / (i + 1)])
                integrated_coefs = tf.tensor_scatter_nd_sub(integrated_coefs, [[i - 1]], [0.5 * coefs[i] / (i - 1)])
        return integrated_coefs

    def create_basic_batch_coefs_fast(self, max_degree: int = None, min_degree: int = None) -> tf.Tensor:
        max_degree = self.max_degree if max_degree is None else max_degree
        min_degree = self.min_degree if min_degree is None else min_degree
        coefs = tf.random.normal(shape=(self.batch_size, max_degree))

        if max_degree == min_degree:
            mask_coefs = tf.ones((self.batch_size, max_degree))
        else:
            mask_coefs = self.create_mask_coefs(max_degree, min_degree)
        coefs = coefs * tf.cast(mask_coefs, dtype=coefs.dtype)
        normalized_values = tf.random.normal(
            shape=(self.batch_size, 1),
            mean=self.norm_mean,
            stddev=self.norm_std,
            dtype=tf.float32
        )
        row_norms = tf.expand_dims(tf.norm(coefs, ord=2, axis=1), axis=1)
        return coefs * (normalized_values / row_norms)

    def create_mask_coefs(self, max_degree: int = None, min_degree: int = None) -> tf.Tensor:
        # min_degree not really supported
        max_degree = self.max_degree if max_degree is None else max_degree
        min_degree = self.min_degree if min_degree is None else min_degree

        # Given set of vectors
        vector_set = self.generate_vector_set(min_degree, max_degree)

        num_classes = vector_set.shape[0]  # Number of available classes (indices)
        sampling_probs = tf.ones((self.batch_size, num_classes)) / num_classes
        logits = tf.math.log(sampling_probs)
        sampled_indices = tf.random.categorical(logits, num_samples=1)
        sampled_indices = tf.squeeze(sampled_indices, axis=-1)
        selected_vectors = tf.gather(vector_set, sampled_indices)

        return selected_vectors

    def generate_vector_set(self, min_degree, max_degree):
        vector_set = []

        for degree in range(min_degree, max_degree + 1):
            vector = [1] * degree + [0] * (max_degree - degree)
            vector_set.append(vector)

        return tf.constant(vector_set, dtype=tf.float32)

    def create_basic_batch_coefs(self, max_degree: int = None, min_degree: int = None) -> tf.Tensor:
        max_degree = self.max_degree if max_degree is None else max_degree
        min_degree = self.min_degree if min_degree is None else min_degree
        degrees = tf.random.uniform(shape=(self.batch_size,), minval=min_degree, maxval=max_degree + 1,
                                    dtype=tf.dtypes.int32)
        degrees_float = tf.cast(degrees, dtype=tf.float32)
        coefs = tf.map_fn(lambda d: self.generate_random_sobolove_functions(tf.cast(d, dtype=tf.int32)), degrees_float)
        return tf.squeeze(coefs)

    def generate_random_sobolove_functions(self, degree: int) -> tf.Tensor:
        coefs = tf.random.normal(shape=[degree, 1])
        normalization_to = tf.random.normal(shape=[], mean=self.norm_mean, stddev=self.norm_std)
        # normalization_to = tf.random.uniform(shape=[], minval=0.25, maxval=1.0)
        normalized_coefs = (coefs / tf.norm(coefs, ord=2)) * normalization_to
        padded_normalized_coefs = tf.pad(normalized_coefs,
                                         paddings=[(0, self.max_degree - tf.shape(normalized_coefs)[0]), (0, 0)])

        return tf.transpose(padded_normalized_coefs)

    def __getitem__(self, index) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        coefs = self.create_batch_coefs()
        int_data = self.poly_layer.call([coefs, tf.cast(self.TRAINING_POINTS, coefs.dtype)])

        train_x = self.add_noise(int_data, snr_level=self.snr_level) if self.add_noise_to_input else int_data
        ext_y = self.poly_layer.call([coefs, tf.cast(self.EXTRAPOLATION_POINTS, coefs.dtype)])
        if self.coef_only:
            return train_x, {'coefs': coefs}
        if self.norm_only:
            return train_x, {'norm': tf.expand_dims(tf.linalg.norm(coefs, ord=2, axis=1), axis=1)}

        return train_x, {'int': int_data, 'coefs': coefs, 'ext': ext_y, 'special': coefs}

    def add_noise(self, signal: tf.Tensor, snr_level: float = 35.0) -> tf.Tensor:
        noise = tf.random.normal(shape=signal.shape, stddev=0.01)
        signal_power = tf.norm(signal, ord=2, axis=1) ** 2
        noise_power = tf.norm(noise, ord=2, axis=1) ** 2
        adopted_snr_level = 10 ** (snr_level / 10)
        alpha = tf.sqrt(signal_power / (adopted_snr_level * noise_power))
        return signal + noise * tf.expand_dims(alpha, axis=1)

    def get_one_function(self, deg: int, snr: float = 35):
        if self.create_monotonic:
            # not efficient at all
            coefs = self.create_chebyshev_monotonic_batch_coefs()
            coefs = coefs[0:1]
        else:
            coefs = self.generate_random_sobolove_functions(deg)

        int_data = self.poly_layer.call([coefs, tf.cast(self.TRAINING_POINTS, coefs.dtype)])
        ext_data = self.poly_layer.call([coefs, tf.cast(self.EXTRAPOLATION_POINTS, coefs.dtype)])
        if self.add_noise_to_input:
            int_train = self.add_noise(tf.cast(tf.expand_dims(int_data, axis=0), tf.float32), snr_level=snr)[0]
        else:
            int_train = int_data
        return {'int_clean': int_data.numpy(), 'ext': ext_data.numpy(), 'coef': coefs[0].numpy(),
                'int_train': int_train.numpy(), 'special': coefs[0].numpy()}


class AnalyticNN(tf.keras.Model):
    def __init__(self, special: bool = False, num_layers: int = 20, num_out: int = 30, inner_width: int = 20):
        super().__init__()
        self.special = special
        if self.special:
            self.dense_layers = [MultiActivationDense(100, activations=[lambda x: x,
                                                                        tf.nn.relu,
                                                                        tf.nn.tanh,
                                                                        tf.sin,
                                                                        tf.cos,
                                                                        lambda x: tf.sin(3 * x),
                                                                        lambda x: tf.cos(3 * x),
                                                                        lambda x: tf.sin(2 * x),
                                                                        lambda x: tf.cos(2 * x)]) for _ in
                                 range(num_layers)]
        else:
            print(inner_width)
            self.dense_layers = [tf.keras.layers.Dense(inner_width, activation=tf.nn.relu) for _ in range(num_layers)]
        self.dense_out = tf.keras.layers.Dense(num_out, activation=None)

    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        return self.dense_out(x)


class ExtrapolationCoefNN(tf.keras.Model):
    def __init__(self, num_layers: int = 2, num_out: int = 30, inner_layer_width: int = 20,
                 polynomial_type: str = "chebyshev", generator: FunctionCoefGenerator = None):
        super().__init__()
        self.dense_layers = [tf.keras.layers.Dense(inner_layer_width, activation=tf.nn.tanh) for _ in
                             range(num_layers)]
        self.dense_out = tf.keras.layers.Dense(num_out, activation=None)
        self.generator = FunctionCoefGenerator(polynomial_type=polynomial_type) if generator is None else generator
        self.poly_layer = generator.poly_layer
        self.coefs = None

    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        x = self.dense_out(x)
        self.coefs = x
        int_points = self.poly_layer.call(
            [self.coefs, tf.cast(self.generator.TRAINING_POINTS, self.coefs.dtype)])
        ext_points = self.poly_layer.call([self.coefs, tf.cast(self.generator.EXTRAPOLATION_POINTS, self.coefs.dtype)])
        return {'int': int_points, 'coefs': self.coefs, 'ext': ext_points, 'special': self.coefs}

    @staticmethod
    def split_income_data_to_step(data):
        if len(data) == 3:
            return data[0], data[1], data[2]
        else:
            return data[0], data[1], None


def create_nn(learning_rate: float = 0.0001, num_layers: int = 20, num_out: int = 30, inner_width: int = 20):
    model = AnalyticNN(num_layers=num_layers, num_out=num_out, inner_width=inner_width)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=EXTRAPOLATION_METRICS)
    model.run_eagerly = False
    return model


class InfiniteExtrapolationPointMSELoss(tf.keras.losses.Loss):
    def __init__(self,
                 start_point: float,
                 end_point: float,
                 polynomial_type: str = None,
                 basis_len: int = 8,
                 basis_functions: List[Callable] = None,
                 second_dimension_start_point: float = 0,
                 second_dimension_end_point: float = 0,
                 **kwargs):
        super().__init__(**kwargs)
        self.polynomial_type = polynomial_type
        self.start_point = start_point
        self.end_point = end_point
        self.second_dimension_start_point = second_dimension_start_point
        self.second_dimension_end_point = second_dimension_end_point
        self.basis_len = basis_len
        self.basis_functions = basis_functions
        if self.polynomial_type == "two_dimensions":
            self.coef_weight_matrix = self.create_coef_weight_matrix_two_dimension()
        else:
            self.coef_weight_matrix = self.create_coef_weight_matrix_one_dimension()

    def call(self, y_true, y_pred):
        # Calculate the element-wise difference between y_true and y_pred
        difference = tf.subtract(y_true, y_pred)

        diff_expanded_outer = tf.expand_dims(difference, axis=2)
        diff_expanded_inner = tf.expand_dims(difference, axis=1)

        # Perform element-wise multiplication and broadcasting
        weighted_outer_product = diff_expanded_outer * diff_expanded_inner * self.coef_weight_matrix

        return tf.reduce_sum(weighted_outer_product, axis=(1, 2))

    def create_coef_weight_matrix_one_dimension(self):
        self.basis_functions
        basis_len = len(self.basis_functions)
        integral_matrix = np.zeros((basis_len, basis_len))
        for i in range(basis_len):
            for j in range(basis_len):
                integrand = lambda x: (self.basis_functions[i](x) * self.basis_functions[j](x)) / (
                        self.end_point - self.start_point)
                integral_matrix[i, j], _ = quad(integrand, self.start_point, self.end_point)

        return tf.convert_to_tensor(integral_matrix, tf.float32)

    def create_coef_weight_matrix_two_dimension(self):
        basis_len = len(self.basis_functions)
        integral_matrix = np.zeros((basis_len, basis_len))
        for i in range(basis_len):
            for j in range(basis_len):
                integrand = self.create_basis_integrad_for_two_dimensional_basis(i, j)
                integral_matrix[i, j], _ = dblquad(func=integrand,
                                                   a=self.start_point,
                                                   b=self.end_point,
                                                   # opposite because we are extrapolating the upper half
                                                   gfun=self.second_dimension_end_point,
                                                   hfun=self.second_dimension_start_point,
                                                   # gfun=np.pi/4,
                                                   # hfun=np.pi,
                                                   )
        # 17.11111 np.pi/2
        # 41.027 np.pi / 3
        # 88.29 np.pi / 4
        return tf.convert_to_tensor(integral_matrix, tf.float32)

    def calculate_condition_number(self):
        basis_len = len(self.basis_functions)
        integral_matrix = np.zeros((basis_len, basis_len))
        for i in range(basis_len):
            for j in range(basis_len):
                if j != i:
                    continue
                integrand = self.create_basis_integrad_for_two_dimensional_basis(i, j)
                integral_matrix[i, j], _ = dblquad(func=integrand,
                                                   a=self.start_point,
                                                   b=self.end_point,
                                                   # opposite because we are extrapolating the upper half
                                                   # gfun=self.second_dimension_end_point,
                                                   # hfun=self.second_dimension_start_point,
                                                   gfun=3 * np.pi / 4,
                                                   hfun=np.pi,
                                                   )

    def prepare_x_y_for_two_dimensional_basis_functions(self, x, y):
        return tf.convert_to_tensor([[x, y]])

    def create_basis_integrad_for_two_dimensional_basis(self, i, j):
        def integrand(x, y):
            t = self.prepare_x_y_for_two_dimensional_basis_functions(x, y)
            return (self.basis_functions[i](self.prepare_x_y_for_two_dimensional_basis_functions(x, y)) *
                    self.basis_functions[j](self.prepare_x_y_for_two_dimensional_basis_functions(x, y)))

        return integrand


def create_nn_coef(learning_rate: float = 0.0001, num_layers: int = 20, num_out: int = 30, mono_lr: float = 0.0001,
                   generator: FunctionCoefGenerator = None,
                   special_loss: InfiniteExtrapolationPointMSELoss = None):
    model = ExtrapolationCoefNN(num_layers=num_layers, num_out=num_out,
                                inner_layer_width=100,
                                generator=generator)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt,
                  loss={'coefs': tf.keras.losses.MeanSquaredError(),
                        'int': tf.keras.losses.MeanSquaredError(),
                        'ext': tf.keras.losses.MeanSquaredError(),
                        "special": special_loss},
                  metrics=EXTRAPOLATION_METRICS,
                  loss_weights={"int": 0.0, "coefs": 0.0, "ext": 0.0, "special": 1.0})
    model.run_eagerly = False
    return model


def create_callbacks(val_monitor_name: str, model_save_path: str) -> Tuple[
    tf.keras.callbacks.EarlyStopping, tf.keras.callbacks.ModelCheckpoint]:
    early = tf.keras.callbacks.EarlyStopping(
        monitor=val_monitor_name,
        min_delta=0,
        patience=1000,
        verbose=0,
        mode="min",
        baseline=None,
        restore_best_weights=True,
    )
    saver = tf.keras.callbacks.ModelCheckpoint(
        model_save_path,
        save_weights_only=True,
        save_best_only=True,
        monitor=val_monitor_name
    )
    return early, saver


def reset_seeds():
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(37)
    rn.seed(1254)
    tf.random.set_seed(89)


class EvalCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        tf.keras.backend.clear_session()

    def on_epoch_end(self, epoch: int, logs=None):
        # Housekeeping
        tf.print("ON END1")
        tf.print(gc.collect())
        tf.print(gc.collect())
        tf.print(gc.collect())

        tf.print("ON END2")


class LearningRateDecay(Callback):
    def __init__(self, decay_rate: float = 0.1, decay_each_n_epochs: int = 1000,
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


def calc_function_points(generator, coefs, num_points):
    if coefs.shape[0] != 1:
        coefs = tf.expand_dims(coefs, axis=0)
    ext_points = generator.poly_layer.call(
        [coefs,
         tf.cast(np.linspace(generator.extrapolation_start_point, generator.end_point, num_points), coefs.dtype)])

    int_points = generator.poly_layer.call(
        [coefs,
         tf.cast(np.linspace(generator.start_point, generator.extrapolation_start_point, num_points), coefs.dtype)])

    return int_points, ext_points


def plot_coef_predictions(real_coefs, pred_coefs, generator, name="None", plot_name_addition: str = "None") -> float:
    pred_int, pred_ext = calc_function_points(generator, pred_coefs, 100)
    int_data, ext_data = calc_function_points(generator, real_coefs, 100)

    data_x = np.linspace(generator.start_point, generator.end_point, 1000)
    data_y = generator.poly_layer.call(
        [tf.expand_dims(real_coefs, axis=0),
         tf.cast(np.linspace(generator.start_point, generator.end_point, 1000), real_coefs.dtype)])
    plt.plot(data_x, data_y[0], color='black', label='The Function', linewidth=2.0)
    # plt.scatter(generator.TRAINING_POINTS, int_data, color='blue', label='Training Data')
    plt.scatter(generator.TRAINING_POINTS, pred_int, color='orange', label=f'Training {name}')
    # plt.scatter(generator.EXTRAPOLATION_POINTS, ext_data, color='green', label='Extrapolation Data')
    plt.scatter(generator.EXTRAPOLATION_POINTS, pred_ext, color='red', label=f'Extrapolation {name}')
    plt.rc('legend', fontsize=13)
    tick_size = 13.5
    plt.rc('xtick', labelsize=tick_size)
    plt.rc('ytick', labelsize=tick_size)

    plt.legend()
    plt.savefig(
        fr"C:\git\mainfold_approximation\results\paper_graphs\{name}_function_{plot_name_addition}.png")
    plt.clf()
    a = 1
    return stateless_rmse(pred_ext, ext_data)


def train_next_chebyshev_functions(max_deg: int, save_name: str, batch_size: int = 25, norm: float = 1.0,
                                   std: float = 0.25, monotonic: bool = False):
    generator = FunctionCoefGenerator(batch_size=batch_size,
                                      max_degree=max_deg,
                                      min_degree=1,
                                      polynomial_type="chebyshev",
                                      add_noise_to_input=True,
                                      norm_mean=norm,
                                      norm_std=std,
                                      create_monotonic=monotonic)

    reset_seeds()
    model = ExtrapolationCoefNN(num_layers=10, num_out=max_deg,
                                inner_layer_width=100,
                                generator=generator)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt,
                  loss={
                      "ext": tf.keras.losses.MeanSquaredError(),
                      "coefs": tf.keras.losses.MeanSquaredError(),
                      "special": InfiniteExtrapolationPointMSELoss(
                          start_point=generator.extrapolation_start_point,
                          end_point=generator.end_point,
                          basis_len=max_deg,
                          polynomial_type=generator.polynomial_type,
                          basis_functions=generator.poly_layer.create_poly_basis_functions(max_deg))
                  },
                  metrics=reset_metrics(EXTRAPOLATION_METRICS),
                  loss_weights={"int": 0.0, "coefs": 0.0, "ext": 0.0, "special": 1.0})
    model.run_eagerly = False
    history = model.fit(generator,
                        epochs=2500 if monotonic else 3400,
                        batch_size=batch_size,
                        callbacks=[LearningRateDecay(decay_each_n_epochs=1000, minimum_learning_rate=0.00001)],
                        verbose=2)

    summarize_diagnostics(history,
                          name=f'./results/chebyshev/{save_name}_1.png',
                          metrics=[metric for metric in history.history if
                                   metric in ['loss', 'special_loss', 'coefs_loss', 'ext_root_mean_squared_error',
                                              'int_root_mean_squared_error', 'coef_root_mean_squared_error']],
                          no_val=True,
                          exclude_first=0)
    summarize_diagnostics(history,
                          name=f'./results/chebyshev/{save_name}_2.png',
                          metrics=[metric for metric in history.history if
                                   metric in ['loss', 'special_loss', 'coefs_loss', 'ext_root_mean_squared_error',
                                              'int_root_mean_squared_error', 'coef_root_mean_squared_error']],
                          no_val=True,
                          exclude_first=200)

    model.save_weights(f"./models/chebyshev/{save_name}.h5")


def create_num_input_data_set():
    # Creates num input point data set
    max_deg = 15
    num_input_points_list = [12, 13, 14, 15, 16, 17, 20, 22, 24]
    for num_input_points in num_input_points_list:
        print(num_input_points)
        data = {}
        generator = FunctionCoefGenerator(batch_size=25, max_degree=max_deg, create_monotonic=False,
                                          polynomial_type="chebyshev",
                                          min_degree=max_deg,
                                          add_noise_to_input=False,
                                          num_input_points=num_input_points)
        for i in range(100):
            one_function_dict = generator.get_one_function(max_deg)
            data[i] = one_function_dict
        df = pd.DataFrame.from_dict(data).transpose()
        df.to_pickle(
            fr"C:\git\mainfold_approximation\dataset\coef_extrapolation\chevi\num_input_point_{num_input_points}_deg_{max_deg}_norm_regular.pickle")
    print("Done")


def create_different_noise_data_set(deg: int = 5, create_no_noise_data_set: bool = False):
    snr_levels = ["no_noise"] if create_no_noise_data_set else [50, 40, 30, 20]
    reset_seeds()
    for snr in snr_levels:
        if snr == 35:
            print("no snr 35 creation")
            raise
        data = {}
        generator = FunctionCoefGenerator(batch_size=25, max_degree=deg, create_monotonic=False,
                                          polynomial_type="chebyshev", add_noise_to_input=False,
                                          norm_mean=1,
                                          norm_std=0.25,
                                          min_degree=deg - 1)
        for i in range(100):
            one_function_dict = generator.get_one_function(deg, snr)
            coefs = tf.expand_dims(one_function_dict['coef'], axis=0)
            int_data = one_function_dict['int_train']
            if not create_no_noise_data_set:
                int_train = generator.add_noise(tf.cast(tf.expand_dims(int_data, axis=0), tf.float32), snr_level=snr)[
                    0].numpy()
            else:
                int_train = int_data
            data[i] = {'int_clean': int_data, 'ext': one_function_dict['ext'], 'coef': coefs[0].numpy(),
                       'int_train': int_train, 'special': coefs[0].numpy()}

        df = pd.DataFrame.from_dict(data).transpose()
        df.to_pickle(
            r"C:\git\mainfold_approximation\dataset\coef_extrapolation\chebyshev\test_data_noise_" + f"{deg}_snr_{snr}_norm_regular.pickle")
    print("Done")


def create_global_data_set(snr: int = 35, create_no_noise_data_set: bool = False):
    reset_seeds()
    for deg in [3, 5, 7]:
        print(deg)
        data = {}
        generator = FunctionCoefGenerator(batch_size=25, max_degree=deg, create_monotonic=False,
                                          polynomial_type="chebyshev",
                                          add_noise_to_input=False,  # adds noise later on
                                          norm_mean=1,
                                          norm_std=0.25,
                                          min_degree=deg - 1)
        for i in range(100):
            one_function_dict = generator.get_one_function(deg, snr)
            coefs = tf.expand_dims(one_function_dict['coef'], axis=0)
            int_data = one_function_dict['int_train']
            if not create_no_noise_data_set:
                int_train = generator.add_noise(tf.cast(tf.expand_dims(int_data, axis=0), tf.float32), snr_level=snr)[
                    0].numpy()
            else:
                int_train = int_data
            data[i] = {'int_clean': int_data, 'ext': one_function_dict['ext'], 'coef': coefs[0].numpy(),
                       'int_train': int_train, 'special': coefs[0].numpy()}

        df = pd.DataFrame.from_dict(data).transpose()
        df.to_pickle(
            r"C:\git\mainfold_approximation\dataset\coef_extrapolation\chebyshev\test_data_noise_" + f"{deg}_snr_{snr}_norm_regular.pickle")
    print("Done")


def test_model_on_chebyshev_data_set(model_weight_path: str, data_set_path: str, model_name: str,
                                     max_deg: int = 7, deg: int = 7, monotonic: bool = False):
    results = {}
    generator = FunctionCoefGenerator(batch_size=25, max_degree=max_deg, create_monotonic=monotonic,
                                      polynomial_type="chebyshev",
                                      min_degree=max_deg,
                                      add_noise_to_input=False,
                                      num_input_points=100)
    # results = {}
    model_test = ExtrapolationCoefNN(num_layers=10, num_out=max_deg,
                                     inner_layer_width=100,
                                     generator=generator)
    dummy_input = tf.ones((1, 100))
    model_test(dummy_input)
    model_test.load_weights(model_weight_path)
    metrics = [stateless_rmse, stateless_mae]
    df = pd.read_pickle(data_set_path)
    max_rmse = 0
    results_list = []
    for i in range(df.shape[0]):
        int_train = df['int_train'][i]
        coef_data = df['coef'][i]
        pred = model_test.predict(int_train)
        pred_coefs = pred['coefs']

        pred_int, pred_ext = calc_function_points(generator, pred_coefs, 100)
        int_data, ext_data = calc_function_points(generator, coef_data, 100)

        res_old = {f"ext_full_{metric.__name__}": metric(ext_data, pred_ext) for metric in
                   metrics}
        coef_data = pad_coef_array(coef_data, deg)
        res_old['coefs_root_mean_squared_error'] = stateless_rmse(coef_data[:deg],
                                                                  pred_coefs[0][:deg])
        res_old['int_root_mean_squared_error_clean'] = stateless_rmse(int_data, pred_int)
        current_rmse = stateless_rmse(pred_ext, ext_data)
        if max_rmse < current_rmse:
            max_rmse = current_rmse

        results_list.append(res_old)

    average_results = average_dicts(results_list)
    results[f"deg_{deg}_int_0_is_coef_0_ext_0_special_1_non_monotonic"] = average_results
    df_results = pd.DataFrame(results).transpose()
    df_results.to_csv(
        f"./results/chebyshev/chebyshev_only_max_deg_{max_deg}_y_reg_{deg}_prediction_monotonic_{monotonic}_{model_name}.csv")
    print("Done")


def test_ls_on_chebyshev_data_set(data_set_path: str = None,
                                  deg_to_predict: int = 7,
                                  deg_data_set: int = 7,
                                  name: str = "snr_35"):
    results = {}
    generator = FunctionCoefGenerator(batch_size=25,
                                      max_degree=deg_to_predict,
                                      min_degree=deg_to_predict,
                                      polynomial_type="chebyshev",
                                      add_noise_to_input=True,
                                      norm_mean=1,
                                      norm_std=0.25)
    metrics = [stateless_rmse, stateless_mae]
    max_rmse = 0
    results_list = []
    df = pd.read_pickle(data_set_path)
    for i in range(df.shape[0]):
        print(i)
        int_train = df['int_train'][i]
        coef_data = df['coef'][i]
        pred_coefs = tf.expand_dims(chebfit(generator.TRAINING_POINTS.numpy(), int_train[0], deg_to_predict - 1),
                                    axis=0)

        pred_int, pred_ext = calc_function_points(generator, pred_coefs, 100)
        int_data, ext_data = calc_function_points(generator, coef_data, 100)

        res_old = {f"ext_full_{metric.__name__}": metric(ext_data, pred_ext) for metric in
                   metrics}

        coef_data = pad_coef_array(coef_data, deg_to_predict)
        res_old['coefs_root_mean_squared_error'] = stateless_rmse(coef_data[:deg_to_predict],
                                                                  pred_coefs[0][:deg_to_predict])
        res_old['int_root_mean_squared_error_clean'] = stateless_rmse(int_data, pred_int)
        current_rmse = stateless_rmse(pred_ext, ext_data)
        if max_rmse < current_rmse:
            max_rmse = current_rmse

        results_list.append(res_old)

    average_results = average_dicts(results_list)
    results[f"baseline_{deg_data_set}"] = average_results
    df_results = pd.DataFrame(results).transpose()
    df_results.to_csv(f"./results/chebyshev/chebyshev_only_max_deg_{deg_data_set}_baseline_{name}.csv")
    print("Done")


def pad_coef_array(coef: np.ndarray, coef_desired_length: int):
    padding_size = max(0, coef_desired_length - len(coef))
    return np.pad(coef, (0, padding_size), mode='constant', constant_values=0)


def test_model_on_chebyshev_data_set_on_one_function(model_weight_path: str, data_set_path: str,
                                                     max_deg: int = 7, monotonic: bool = False,
                                                     function_index: int = 7):
    generator = FunctionCoefGenerator(batch_size=25, max_degree=max_deg, create_monotonic=monotonic,
                                      polynomial_type="chebyshev",
                                      min_degree=max_deg,
                                      add_noise_to_input=False,
                                      num_input_points=100)

    model_test = ExtrapolationCoefNN(num_layers=10, num_out=max_deg,
                                     inner_layer_width=100,
                                     generator=generator)
    dummy_input = tf.ones((1, 100))
    model_test(dummy_input)
    model_test.load_weights(model_weight_path)
    df = pd.read_pickle(data_set_path)
    int_train = df['int_train'][function_index]
    real_coefs = df['coef'][function_index]
    pred = model_test.predict(int_train)
    pred_coefs = pred['coefs']

    if monotonic:
        rmse = plot_coef_predictions(real_coefs, pred_coefs, generator, name="NExT Monotonic",
                                     plot_name_addition="monotone")
    else:
        rmse = plot_coef_predictions(real_coefs, pred_coefs, generator, name="NExT", plot_name_addition="general")
    print(rmse)


def test_ls_on_chebyshev_data_set_on_one_function(data_set_path: str = None,
                                                  deg_to_predict: int = 7,
                                                  function_index: int = 7,
                                                  plot_name_addition: str = "None"):
    generator = FunctionCoefGenerator(batch_size=25,
                                      max_degree=deg_to_predict,
                                      min_degree=deg_to_predict,
                                      polynomial_type="chebyshev",
                                      add_noise_to_input=True,
                                      norm_mean=1,
                                      norm_std=0.25)

    df = pd.read_pickle(data_set_path)

    int_train = df['int_train'][function_index]
    real_coefs = df['coef'][function_index]
    pred_coefs = tf.expand_dims(chebfit(generator.TRAINING_POINTS.numpy(), int_train[0], deg_to_predict - 1),
                                axis=0)

    rmse = plot_coef_predictions(real_coefs, pred_coefs, generator, name="LS", plot_name_addition=plot_name_addition)
    print(rmse)


def test_ls_on_different_noise_chebyshev_data_sets():
    for noise_level in ["no_noise", 50, 40, 30, 20]:
        data_set_path = r"C:\git\mainfold_approximation\dataset\coef_extrapolation\chebyshev\test_data_noise_" + f"5_snr_{noise_level}_norm_regular.pickle"
        test_ls_on_chebyshev_data_set(data_set_path=data_set_path, deg_to_predict=7, deg_data_set=5,
                                      name=f"snr_{noise_level}")


def test_model_on_different_noise_chebyshev_data_sets():
    for noise_level in ["no_noise", 50, 40, 30, 20]:
        data_set_path = r"C:\git\mainfold_approximation\dataset\coef_extrapolation\chebyshev\test_data_noise_" + f"5_snr_{noise_level}_norm_regular.pickle"
        test_model_on_chebyshev_data_set(model_weight_path=f"./models/chebyshev/chebyshev_max_deg_7_norm_1_std_0.25.h5",
                                         data_set_path=data_set_path,
                                         model_name=f"chebyshev_max_deg_7_norm_1_std_0.25_snr_{noise_level}",
                                         max_deg=7,
                                         deg=5,
                                         monotonic=False)


def test_ls_on_global_chebyshev_data_sets():
    for deg in [3, 5, 7]:
        data_set_path = r"C:\git\mainfold_approximation\dataset\coef_extrapolation\chebyshev\test_data_noise_" + \
                        f"{deg}_snr_35_norm_regular.pickle"
        test_ls_on_chebyshev_data_set(data_set_path=data_set_path, deg_to_predict=7, deg_data_set=deg,
                                      name=f"snr_35")


def test_model_on_global_chebyshev_data_sets():
    for deg in [3, 5, 7]:
        data_set_path = r"C:\git\mainfold_approximation\dataset\coef_extrapolation\chebyshev\test_data_noise_" + \
                        f"{deg}_snr_35_norm_regular.pickle"
        test_model_on_chebyshev_data_set(model_weight_path=f"./models/chebyshev/chebyshev_max_deg_7_norm_1_std_0.25.h5",
                                         data_set_path=data_set_path,
                                         model_name="chebyshev_max_deg_7_norm_1_std_0.25",
                                         max_deg=7,
                                         deg=deg,
                                         monotonic=False)


def main():
    function_index = 4  # monotonic function graph index - 7 degree
    print(function_index)
    test_model_on_chebyshev_data_set_on_one_function(
        f"./models/chebyshev/chebyshev_max_deg_7_norm_1_std_0.25_monotonic.h5",
        r"C:\git\mainfold_approximation\dataset\coef_extrapolation\chebyshev\test_data_noise_" +
        f"7_snr_35_norm_regular_monotonic.pickle",
        monotonic=True,
        function_index=function_index)

    test_ls_on_chebyshev_data_set_on_one_function(
        data_set_path=r"C:\git\mainfold_approximation\dataset\coef_extrapolation\chebyshev\test_data_noise_" +
                      f"7_snr_35_norm_regular_monotonic.pickle",
        function_index=function_index,
        plot_name_addition="monotonic")

    function_index = 48  # general function graph index - 5 degree
    test_model_on_chebyshev_data_set_on_one_function(
        f"./models/chebyshev/chebyshev_max_deg_7_norm_1_std_0.25.h5",
        r"C:\git\mainfold_approximation\dataset\coef_extrapolation\chebyshev\test_data_noise_" +
        f"5_snr_35_norm_regular.pickle",
        monotonic=False,
        function_index=function_index)

    test_ls_on_chebyshev_data_set_on_one_function(
        data_set_path=r"C:\git\mainfold_approximation\dataset\coef_extrapolation\chebyshev\test_data_noise_" +
                      f"5_snr_35_norm_regular.pickle",
        function_index=function_index,
        plot_name_addition="general")
    train_next_chebyshev_functions(max_deg=7,
                                   save_name="chebyshev_max_deg_7_norm_1_std_0.25_monotonic",
                                   norm=1.0,
                                   std=0.25,
                                   monotonic=True)

    test_model_on_chebyshev_data_set(
        model_weight_path=f"./models/chebyshev/chebyshev_max_deg_7_norm_1_std_0.25_monotonic.h5",
        data_set_path=r"C:\git\mainfold_approximation\dataset\coef_extrapolation\chebyshev\test_data_noise_" +
                      f"3_snr_35_norm_regular_monotonic.pickle",
        model_name="chebyshev_max_deg_7_norm_1_std_0.25_monotonic8888888",
        deg=3,
        monotonic=True)
    test_model_on_chebyshev_data_set(
        model_weight_path=f"./models/chebyshev/chebyshev_max_deg_7_norm_1_std_0.25_monotonic.h5",
        data_set_path=r"C:\git\mainfold_approximation\dataset\coef_extrapolation\chebyshev\test_data_noise_" +
                      f"5_snr_35_norm_regular_monotonic.pickle",
        model_name="chebyshev_max_deg_7_norm_1_std_0.25_monotonic",
        deg=5,
        monotonic=True)
    test_model_on_chebyshev_data_set(
        model_weight_path=f"./models/chebyshev/chebyshev_max_deg_7_norm_1_std_0.25_monotonic.h5",
        data_set_path=r"C:\git\mainfold_approximation\dataset\coef_extrapolation\chebyshev\test_data_noise_" +
                      f"7_snr_35_norm_regular_monotonic.pickle",
        model_name="chebyshev_max_deg_7_norm_1_std_0.25_monotonic",
        deg=7,
        monotonic=True)

    # create_different_noise_data_set()
    # create_different_noise_data_set(create_no_noise_data_set=True)
    test_ls_on_different_noise_chebyshev_data_sets()
    test_model_on_different_noise_chebyshev_data_sets()


if __name__ == "__main__":
    main()
