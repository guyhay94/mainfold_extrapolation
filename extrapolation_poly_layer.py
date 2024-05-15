import numpy as np
import tensorflow as tf
from functools import partial
from typing import Callable, List, Tuple
from scipy.special import sph_harm

from find_simple_baselines import create_fourier_basis_functions_tf, create_chebyshev_basis_functions


def create_tf_nth_non_basis_polynomial(polynomial_basis: List[Callable]):
    def tf_nth_non_basis_polynomial(x, degree):
        return polynomial_basis[degree](x)
    return tf_nth_non_basis_polynomial

def tf_nth_trigonometric_polynomial(x, degree):
    # n=0 -> 1
    # n%2==1 -> sin(n//2 x)
    # n%2==0 -> cos(n//2 x)
    if degree == 0:
        return tf.ones_like(x)
    elif degree % 2 == 1:
        return tf.sin(((degree + 1) // 2) * x)
    else:
        return tf.cos(((degree + 1) // 2) * x)

def tf_nth_chebyshev_polynomial(x, degree):
    # T_(n+1)=2xT_n-T_(n-1)
    if degree == 0:
        return tf.ones_like(x)
    elif degree == 1:
        return x
    else:
        return 2 * x * tf_nth_chebyshev_polynomial(x, degree - 1) - tf_nth_chebyshev_polynomial(x, degree - 2)


def tf_nth_legendre_polynomial(x, degree):
    # (n+1)P_(n+1)=(2n+1)xP_n-nP_(n-1)
    if degree == 0:
        return tf.ones_like(x)
    elif degree == 1:
        return x
    else:
        return ((2 * degree - 1) * x * tf_nth_legendre_polynomial(x, degree - 1) - (
                degree - 1) * tf_nth_legendre_polynomial(x, degree - 2)) / degree

def real_spherical_harmonics(x: tf.Tensor, l: int, m: int) -> float:
    if m == 0:
        output = sph_harm(m, l, x[:, 0], x[:, 1])
    elif m < 0:
        output = 1/(1j * 2 ** 0.5) * (sph_harm(-m, l, x[:, 0], x[:, 1]) - (-1)**m * sph_harm(m, l, x[:, 0], x[:, 1]))
    else:
        output = 1 / (2 ** 0.5) * (sph_harm(m, l, x[:, 0], x[:, 1]) + (-1) ** m * sph_harm(-m, l, x[:, 0], x[:, 1]))
    return tf.math.real(output)

def real_spherical_harmonics_ordered(x: tf.Tensor, degree: int) -> float:
    l, m = generate_spherical_harmonics_l_m_from_degree(degree)
    return real_spherical_harmonics(x, l, m)

def generate_spherical_harmonics_l_m_from_degree(degree: int) -> Tuple[int, int]:
    l = generate_spherical_harmonics_l_from_degree(degree)
    m = generate_spherical_harmonics_m_from_degree_and_l(degree, l)
    return l, m

def generate_spherical_harmonics_l_from_degree(degree: int, deviation: int = 3) -> int:
    if degree <= 0:
        return 0
    if 1 <= degree <= 3:
        return 1
    return generate_spherical_harmonics_l_from_degree(degree - deviation, deviation+2) + 1

def generate_spherical_harmonics_m_from_degree_and_l(degree: int, l: int) -> int:
    starting_degree = l**2 # easy proof by injunction
    degree_in_degree = degree - starting_degree
    return l - degree_in_degree

def create_spherical_harmonics_basis_element(degree: int = 7):
    l, m = generate_spherical_harmonics_l_m_from_degree(degree)
    return lambda x: real_spherical_harmonics(x, l, m)

def create_spherical_harmonics_functions(deg: int = 7):
    return [create_spherical_harmonics_basis_element(d) for d in range(deg)]

def apply_polynomial(coeffs: tf.Tensor, x: tf.Tensor, nth_poly_func: Callable) -> tf.Tensor:
    # coeffs[n-1] + coeffs[n-2] * T_1 + ... + coeffs[0] * T_(n-1)
    return tf.reduce_sum([tf.expand_dims(coeffs[:, i], axis=1) * tf.cast(nth_poly_func(x, i), np.float32) for i in range(coeffs.shape[1])], axis=0)

tf_chebyshev_val = partial(apply_polynomial, nth_poly_func=tf_nth_chebyshev_polynomial)
tf_legendre_val = partial(apply_polynomial, nth_poly_func=tf_nth_legendre_polynomial)
tf_trigonometric_val = partial(apply_polynomial, nth_poly_func=tf_nth_trigonometric_polynomial)
tf_spherical_harmonics_val = partial(apply_polynomial, nth_poly_func=real_spherical_harmonics_ordered)

class PolynomialCoefLayer:
    def __init__(self, polynomial_type: str, polynomial_basis: List[Callable] = None, name: str = "polynomial_layer"):
        self.__name__ = name
        self.trainable = False
        self.polynomial_type = polynomial_type
        self.polynomial_basis = polynomial_basis
        # For all the polynomial domain is [-1,1]
        if polynomial_type == "polynomial":
            poly_func = tf.math.polyval
        elif polynomial_type == "chebyshev":
            poly_func = tf_chebyshev_val
        elif polynomial_type == "legendre":
            poly_func = tf_legendre_val
        elif polynomial_type == "trigonometric":
            poly_func = tf_trigonometric_val
        elif polynomial_type == "spherical_harmonics":
            poly_func = tf_spherical_harmonics_val
        elif polynomial_type == "non_basis":
            nth_poly_func = create_tf_nth_non_basis_polynomial(polynomial_basis=self.polynomial_basis)
            poly_func = partial(apply_polynomial, nth_poly_func=nth_poly_func)
        else:
            raise NotImplementedError(f"{polynomial_type} not an implemented polynomial function")

        self.poly_func = poly_func

    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        return self.poly_func(*inputs)

    def create_poly_basis_functions(self, basis_len):
        if self.polynomial_type == "trigonometric":
            return create_fourier_basis_functions_tf(basis_len)
        elif self.polynomial_type == "chebyshev":
            return create_chebyshev_basis_functions(basis_len)
        elif self.polynomial_type == "spherical_harmonics":
            return create_spherical_harmonics_functions(basis_len)
        else:
            raise NotImplementedError(
                f"Polynomial type: {self.polynomial_type} is not implemented for InfiniteExtrapolationPointMSELoss")

