import tensorflow as tf
import numpy as np
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
from os.path import isfile, join
from sklearn.datasets import make_spd_matrix

from typing import Tuple, List, Union

dataset_file_names = ["train_x.npy", "train_y.npy",
                      "val_x.npy", "val_y.npy",
                      "test_x.npy", "test_y.npy"]

def tensor_is_almost_equal(x: tf.Tensor, y: tf.Tensor, rtol: float = 1e-14):
    return np.allclose(x.numpy(), y.numpy(), rtol=rtol)


def tensor_eye_like(x: tf.Tensor) -> tf.Tensor:
    return tf.zeros_like(x) + tf.eye(x.shape[-1])

# plot diagnostic learning curves
def summarize_diagnostics(history, metrics: List[str] = ['loss', 'mean_squared_error'],
                          name: str = "model_history", exclude_first: int = 0,
                          no_train: bool = False,
                          no_val: bool = False) -> None:
    fig, axs = plt.subplots(nrows=len(metrics), ncols=1, figsize=(12, 12))
    for index, metric in enumerate(metrics):
        plot = axs[index]
        plot.set_title(f'{metric}')
        if not no_train:
            plot.plot(history.history[metric][exclude_first:], color='blue', label='train')
        if not no_val:
            plot.plot(history.history[f'val_{metric}'][exclude_first:], color='orange', label='val')

    # save plot to file
    plt.tight_layout()
    plt.savefig(name + f'_plot.png')
    plt.close()


def check_losses_input(y_true: tf.Tensor, y_pred: tf.Tensor, name: str = "not specified"):
    error_message = f"y_true, y_pred do not share the same shape, happend in: {name}"
    assert y_true.shape == y_pred.shape, error_message

def visualize_3D(points: np.ndarray):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    plt.show()

def visualize_3D_train_val_test(train_points: np.ndarray, val_points: np.ndarray, test_points: np.ndarray,
                                x_min_lim: Union[float, None] = -1, x_max_lim: Union[float, None] = 1,
                                y_min_lim: Union[float, None] = -1, y_max_lim: Union[float, None] = 1,
                                z_min_lim: Union[float, None] = -1, z_max_lim: Union[float, None] = 1,
                                title="None", save_path: str = None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim(x_min_lim, x_max_lim)
    ax.set_ylim(y_min_lim, y_max_lim)
    ax.set_zlim(z_min_lim, z_max_lim)
    ax.scatter(train_points[:, 0], train_points[:, 1], train_points[:, 2], c='green', label='train set')
    ax.scatter(val_points[:, 0], val_points[:, 1], val_points[:, 2], c='yellow', label='validation set')
    ax.scatter(test_points[:, 0], test_points[:, 1], test_points[:, 2], c='red', label='test set')
    plt.title(title)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    plt.close(fig)


def stateless_rmse(y_true: tf.Tensor, y_pred: tf.Tensor):
    rmse = tf.keras.metrics.RootMeanSquaredError()
    return rmse(y_true, y_pred).numpy()

def stateless_mae(y_true: tf.Tensor, y_pred: tf.Tensor):
    rmse = tf.keras.metrics.MeanAbsoluteError()
    return rmse(y_true, y_pred).numpy()

def reset_metrics(metrics: List[tf.keras.metrics.Metric]) -> List[tf.keras.metrics.Metric]:
    [metric.reset_state() for metric in metrics]
    return metrics

def average_dicts(dict_list):
    # Check if the input list is empty
    if not dict_list:
        return {}

    # Initialize a dictionary to store the average values
    avg_dict = {}

    # Loop through the dictionaries in the list
    for d in dict_list:
        for key, value in d.items():
            # If the key is not in the average dictionary, add it
            if key not in avg_dict:
                avg_dict[key] = value
            else:
                # If the key is already in the average dictionary, accumulate the values
                avg_dict[key] += value

    # Calculate the average by dividing each accumulated value by the number of dictionaries
    num_dicts = len(dict_list)
    for key in avg_dict.keys():
        avg_dict[key] /= num_dicts

    return avg_dict