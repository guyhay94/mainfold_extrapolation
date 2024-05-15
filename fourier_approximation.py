import numpy as np


def an(data: np.ndarray, n: int, x: np.ndarray, period=2 * np.pi):
    # when period=2*np.pi this equals: data*np.cos(nx)
    c = data * np.cos(2 * n * np.pi * x / period)
    return c.sum() / data.shape[0]


def bn(data: np.ndarray, n: int, x: np.ndarray, period=2 * np.pi):
    # when period=2*np.pi this equals: data*np.sin(nx)
    c = data * np.sin(2 * n * np.pi * x / period)
    return c.sum() / data.shape[0]


def a0(data: np.ndarray):
    return data.sum() / data.shape[0]


def create_fourier_function(data: np.ndarray, x: np.ndarray, num_coefs_to_use: int = 3,
                            return_coefs_aswell: bool = False):
    # Building coef array like this
    # n=0 -> 1
    # n%2==1 -> sin(n//2 x)
    # n%2==0 -> cos(n//2 x)
    const = a0(data)
    a_coefs = [an(data, n, x) for n in range(1, num_coefs_to_use + 1)]
    b_coefs = [bn(data, n, x) for n in range(1, num_coefs_to_use + 1)]
    interwined_coefs = np.column_stack((b_coefs, a_coefs)).reshape(-1)
    coefs = np.concatenate([np.array([const]), interwined_coefs])
    def fourier_function(x_to_evaluate):
        a_values = [a_coefs[n] * np.cos((n+1)*x_to_evaluate) for n in range(0, num_coefs_to_use)]
        b_values = [b_coefs[n] * np.sin((n+1)*x_to_evaluate) for n in range(0, num_coefs_to_use)]
        a_summed_value = np.stack(a_values, axis=1).sum(axis=1)
        b_summed_value = np.stack(b_values, axis=1).sum(axis=1)
        return const + a_summed_value + b_summed_value

    if return_coefs_aswell:
        return fourier_function, coefs
    return fourier_function
