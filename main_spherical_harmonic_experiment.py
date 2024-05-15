import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from find_simple_baselines import find_best_coefs_for_specific_basis_functions
from main_function_extrapolation import reset_seeds, FunctionCoefGenerator, ExtrapolationCoefNN, \
    InfiniteExtrapolationPointMSELoss, LearningRateDecay, pad_coef_array
from utils import reset_metrics, summarize_diagnostics, stateless_rmse, stateless_mae, average_dicts

EXTRAPOLATION_METRICS = [tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]


def plot_spherical_harmonics_error(y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   generator: FunctionCoefGenerator,
                                   error_min: float = None,
                                   error_max: float = None,
                                   name: str = "Unknown",
                                   not_error: bool = False):
    tick_size = 13.5
    plt.rc('xtick', labelsize=tick_size)
    plt.rc('ytick', labelsize=tick_size)
    theta = generator.EXTRAPOLATION_POINTS[:, 0].numpy().reshape((100, -1))
    phi = generator.EXTRAPOLATION_POINTS[:, 1].numpy().reshape((100, -1))
    if not not_error:
        errors = np.abs(y_true - y_pred).reshape((100, -1))
    else:
        errors = y_pred.reshape((100, -1))
    error_max = errors.max() if error_max is None else error_max
    error_min = errors.min() if error_min is None else error_min
    X = np.sin(phi) * np.cos(theta)
    Y = np.sin(phi) * np.sin(theta)
    Z = np.cos(phi)

    # cmap = cm.gist_ncar
    cmap = cm.jet

    norm = plt.Normalize(error_min, error_max)
    rgba = cmap(norm(errors))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plot = ax.plot_surface(
        X, Y, Z, rstride=1, cstride=1, facecolors=rgba,
        linewidth=0, antialiased=False, alpha=0.5)

    mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    cbar = plt.colorbar(mappable, ax=ax, pad=0.1)
    ax.view_init(elev=25, azim=130)
    # ax.set_aspect('equal', adjustable='box')
    ax.set_zlim(0, 2)

    plt.savefig(
        fr"C:\git\mainfold_approximation\results\spherical_harmonics\{name}_errors_above_not_error_{not_error}_2.png")

    return errors.min(), errors.max()


def train_ie_spherical_harmonics_functions(max_deg: int, save_name: str, batch_size: int = 25, norm: float = 1.0,
                                           std: float = 0.25,
                                           extrapolation_distance_from_approximation: float = 0):
    generator = FunctionCoefGenerator(batch_size=batch_size,
                                      max_degree=max_deg,
                                      min_degree=1,
                                      polynomial_type="spherical_harmonics",
                                      add_noise_to_input=True,
                                      norm_mean=norm,
                                      norm_std=std,
                                      extrapolation_distance_from_approximation=extrapolation_distance_from_approximation)
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
                          polynomial_type="two_dimensions",
                          basis_functions=generator.poly_layer.create_poly_basis_functions(max_deg),
                          second_dimension_start_point=generator.second_dimension_ext_start_point,
                          second_dimension_end_point=generator.second_dimension_ext_end_point)
                  },
                  metrics=reset_metrics(EXTRAPOLATION_METRICS),
                  loss_weights={"int": 0.0, "coefs": 0.0, "ext": 0.0, "special": 1.0})
    model.run_eagerly = True  # because we use scipy spherical harmonic
    history = model.fit(generator,
                        epochs=3000,
                        batch_size=batch_size,
                        callbacks=[LearningRateDecay(decay_each_n_epochs=1000, minimum_learning_rate=0.00001)],
                        verbose=2)

    summarize_diagnostics(history,
                          name=f'./results/spherical_harmonics/{save_name}_1.png',
                          metrics=[metric for metric in history.history if
                                   metric in ['loss', 'special_loss', 'coefs_loss', 'ext_root_mean_squared_error',
                                              'int_root_mean_squared_error', 'coef_root_mean_squared_error']],
                          no_val=True,
                          exclude_first=0)
    summarize_diagnostics(history,
                          name=f'./results/spherical_harmonics/{save_name}_2.png',
                          metrics=[metric for metric in history.history if
                                   metric in ['loss', 'special_loss', 'coefs_loss', 'ext_root_mean_squared_error',
                                              'int_root_mean_squared_error', 'coef_root_mean_squared_error']],
                          no_val=True,
                          exclude_first=200)

    model.save_weights(f"./models/spherical_harmonics/{save_name}.h5")


def test_ls_on_spherical_harmonics_data_set(data_set_path: str = None,
                                            deg_to_predict: int = 7,
                                            deg_data_set: int = 7,
                                            name: str = "snr_35",
                                            extrapolation_distance_from_approximation: float = 0):
    results = {}
    generator = FunctionCoefGenerator(batch_size=25,
                                      max_degree=deg_to_predict,
                                      min_degree=deg_to_predict,
                                      polynomial_type="spherical_harmonics",
                                      add_noise_to_input=True,
                                      norm_mean=1,
                                      norm_std=0.25,
                                      extrapolation_distance_from_approximation=extrapolation_distance_from_approximation)
    metrics = [stateless_rmse, stateless_mae]
    max_rmse = 0
    results_list = []
    df = pd.read_pickle(data_set_path)
    for i in range(df.shape[0]):
        print(i)
        int_train = df['int_train'][i]
        coef_data = df['coef'][i]
        pred_coefs = find_best_coefs_for_specific_basis_functions(generator.TRAINING_POINTS, int_train,
                                                                  generator.poly_layer.create_poly_basis_functions(
                                                                      deg_to_predict))
        print(coef_data)
        print(pred_coefs)

        pred_int, pred_ext = calc_function_on_train_ext_points(generator, tf.cast(pred_coefs, np.float32))
        int_data, ext_data = calc_function_on_train_ext_points(generator, coef_data)

        res_old = {f"ext_full_{metric.__name__}": metric(ext_data, pred_ext) for metric in
                   metrics}

        coef_data = pad_coef_array(coef_data, deg_to_predict)
        res_old['coefs_root_mean_squared_error'] = stateless_rmse(coef_data[:deg_to_predict],
                                                                  pred_coefs[:deg_to_predict])
        res_old['int_root_mean_squared_error_clean'] = stateless_rmse(int_data, pred_int)
        current_rmse = stateless_rmse(pred_ext, ext_data)
        print(f"ext: {current_rmse}")
        int_rmse = stateless_rmse(int_data, pred_int)
        print(f"int: {int_rmse}")
        if max_rmse < current_rmse:
            max_rmse = current_rmse

        results_list.append(res_old)

    average_results = average_dicts(results_list)
    results[f"baseline_{deg_data_set}"] = average_results
    df_results = pd.DataFrame(results).transpose()
    df_results.to_csv(
        f"./results/spherical_harmonics/spherical_harmonics_only_max_deg_{deg_data_set}_baseline_{name}.csv")
    print("Done")


def test_model_on_spherical_harmonics_data_set(model_weight_path: str, data_set_path: str, model_name: str,
                                               max_deg: int = 7, deg: int = 7,
                                               extrapolation_distance_from_approximation: float = 0):
    results = {}
    generator = FunctionCoefGenerator(batch_size=25, max_degree=max_deg,
                                      polynomial_type="spherical_harmonics",
                                      min_degree=max_deg,
                                      add_noise_to_input=True,
                                      num_input_points=100,
                                      extrapolation_distance_from_approximation=extrapolation_distance_from_approximation)
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
        print(i)
        int_train = df['int_train'][i]
        coef_data = df['coef'][i]
        pred = model_test(int_train)  # can't use predict because of scipy spherical harmonic
        pred_coefs = pred['coefs']

        pred_int, pred_ext = calc_function_on_train_ext_points(generator, pred_coefs)
        int_data, ext_data = calc_function_on_train_ext_points(generator, coef_data)
        res_old = {f"ext_full_{metric.__name__}": metric(ext_data, pred_ext) for metric in
                   metrics}
        coef_data = pad_coef_array(coef_data, deg)
        res_old['coefs_root_mean_squared_error'] = stateless_rmse(coef_data[:deg],
                                                                  pred_coefs[0][:deg])
        res_old['int_root_mean_squared_error_clean'] = stateless_rmse(int_data, pred_int)
        current_rmse = stateless_rmse(pred_ext, ext_data)
        print(current_rmse)
        if max_rmse < current_rmse:
            max_rmse = current_rmse

        results_list.append(res_old)

    average_results = average_dicts(results_list)
    results[f"deg_{deg}_int_0_is_coef_0_ext_0_special_1"] = average_results
    df_results = pd.DataFrame(results).transpose()
    df_results.to_csv(
        f"./results/spherical_harmonics/spherical_harmonics_only_max_deg_{max_deg}_y_reg_{deg}_prediction_{model_name}.csv")
    print("Done")


def calc_function_on_train_ext_points(generator: FunctionCoefGenerator, coefs):
    if coefs.shape[0] != 1:
        coefs = tf.expand_dims(coefs, axis=0)
    ext_points = generator.poly_layer.call([coefs, generator.EXTRAPOLATION_POINTS])

    int_points = generator.poly_layer.call([coefs, generator.TRAINING_POINTS])

    return int_points, ext_points


def create_global_data_set_spherical_harmonics(snr: int = 35, create_no_noise_data_set: bool = False,
                                               deg_to_create: int = 10,
                                               extrapolation_distance_from_approximation: float = 0):
    reset_seeds()
    data = {}
    generator = FunctionCoefGenerator(batch_size=25, max_degree=deg_to_create,
                                      polynomial_type="spherical_harmonics",
                                      add_noise_to_input=False,  # adds noise later on
                                      norm_mean=1,
                                      norm_std=0.25,
                                      min_degree=deg_to_create - 1,
                                      extrapolation_distance_from_approximation=extrapolation_distance_from_approximation)
    for i in range(100):
        print(i)
        one_function_dict = generator.get_one_function(deg_to_create, snr)
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
        r"C:\git\mainfold_approximation\dataset\coef_extrapolation\spherical_harmonics\test_data_noise_" + f"{deg_to_create}_snr_{snr}_norm_regular.pickle")
    print("Done")


def test_model_on_spherical_harmonics_data_set_on_one_function(model_weight_path: str, data_set_path: str,
                                                               max_deg: int = 7,
                                                               function_index: int = 7,
                                                               extrapolation_distance_from_approximation: float = 0,
                                                               error_min: float = None,
                                                               error_max: float = None,
                                                               not_error: bool = False):
    generator = FunctionCoefGenerator(batch_size=25, max_degree=max_deg,
                                      polynomial_type="spherical_harmonics",
                                      min_degree=max_deg,
                                      add_noise_to_input=False,
                                      num_input_points=100,
                                      extrapolation_distance_from_approximation=extrapolation_distance_from_approximation)

    model_test = ExtrapolationCoefNN(num_layers=10, num_out=max_deg,
                                     inner_layer_width=100,
                                     generator=generator)
    dummy_input = tf.ones((1, 100))
    model_test(dummy_input)
    model_test.load_weights(model_weight_path)
    df = pd.read_pickle(data_set_path)
    int_train = df['int_train'][function_index]
    real_coefs = df['coef'][function_index]
    pred = model_test(int_train)
    pred_coefs = pred['coefs']

    pred_int, pred_ext = calc_function_on_train_ext_points(generator, pred_coefs)
    int_data, ext_data = calc_function_on_train_ext_points(generator, real_coefs)

    error_min, error_max = plot_spherical_harmonics_error(ext_data.numpy(), pred_ext.numpy(),
                                                          generator=generator,
                                                          error_min=error_min,
                                                          error_max=error_max,
                                                          name="IE",
                                                          not_error=not_error)
    rmse = stateless_rmse(pred_ext, ext_data)
    print(rmse)
    return error_min, error_max


def plot_spherical_harmonics_data_set_on_one_function(data_set_path: str,
                                                      max_deg: int = 7,
                                                      function_index: int = 7,
                                                      extrapolation_distance_from_approximation: float = 0,
                                                      error_min: float = None,
                                                      error_max: float = None):
    generator = FunctionCoefGenerator(batch_size=25, max_degree=max_deg,
                                      polynomial_type="spherical_harmonics",
                                      min_degree=max_deg,
                                      add_noise_to_input=False,
                                      num_input_points=100,
                                      extrapolation_distance_from_approximation=extrapolation_distance_from_approximation)

    df = pd.read_pickle(data_set_path)
    int_train = df['int_train'][function_index]
    real_coefs = df['coef'][function_index]

    pred_int, pred_ext = calc_function_on_train_ext_points(generator, real_coefs)
    int_data, ext_data = calc_function_on_train_ext_points(generator, real_coefs)
    error_min, error_max = plot_spherical_harmonics_error(ext_data.numpy(), pred_ext.numpy(),
                                                          generator=generator,
                                                          error_min=error_min,
                                                          error_max=error_max,
                                                          name="True",
                                                          not_error=True)
    rmse = stateless_rmse(pred_ext, ext_data)
    print(rmse)
    return error_min, error_max


def test_ls_on_spherical_harmonics_data_set_on_one_function(data_set_path: str = None,
                                                            deg_to_predict: int = 7,
                                                            function_index: int = 7,
                                                            extrapolation_distance_from_approximation: float = 0,
                                                            error_min: float = None,
                                                            error_max: float = None,
                                                            not_error: bool = False):
    generator = FunctionCoefGenerator(batch_size=25,
                                      max_degree=deg_to_predict,
                                      min_degree=deg_to_predict,
                                      polynomial_type="spherical_harmonics",
                                      add_noise_to_input=True,
                                      norm_mean=1,
                                      norm_std=0.25,
                                      extrapolation_distance_from_approximation=extrapolation_distance_from_approximation)

    df = pd.read_pickle(data_set_path)

    int_train = df['int_train'][function_index]
    real_coefs = df['coef'][function_index]
    pred_coefs = find_best_coefs_for_specific_basis_functions(generator.TRAINING_POINTS, int_train,
                                                              generator.poly_layer.create_poly_basis_functions(
                                                                  deg_to_predict))
    print(pred_coefs)
    pred_int, pred_ext = calc_function_on_train_ext_points(generator, tf.cast(pred_coefs, np.float32))
    int_data, ext_data = calc_function_on_train_ext_points(generator, real_coefs)
    error_min, error_max = plot_spherical_harmonics_error(ext_data.numpy(), pred_ext.numpy(),
                                                          generator=generator,
                                                          name="LS",
                                                          not_error=not_error,
                                                          error_min=error_min,
                                                          error_max=error_max)
    rmse = stateless_rmse(pred_ext, ext_data)
    int_rmse = stateless_rmse(int_data, pred_int)
    print(rmse)

    return error_min, error_max


def main():
    deg = 9
    # create_global_data_set_spherical_harmonics(deg_to_create=9,
    #                                            extrapolation_distance_from_approximation=np.pi / 6)
    # create_global_data_set_spherical_harmonics(deg_to_create=5,
    #                                            extrapolation_distance_from_approximation=np.pi / 6)

    train_ie_spherical_harmonics_functions(max_deg=deg,
                                           save_name=f"spherical_harmonics_max_deg_{deg}_norm_1_std_0.25",
                                           norm=1.0,
                                           std=0.25,
                                           extrapolation_distance_from_approximation=np.pi / 6)
    test_model_on_spherical_harmonics_data_set(
        model_weight_path=f"./models/spherical_harmonics/spherical_harmonics_max_deg_9_norm_1_std_0.25.h5",
        data_set_path=r"C:\git\mainfold_approximation\dataset\coef_extrapolation\spherical_harmonics\test_data_noise_" +
                      f"9_snr_35_norm_regular.pickle",
        model_name="spherical_harmonics_max_deg_9_norm_1_std_0.25",
        deg=9,
        max_deg=9,
        extrapolation_distance_from_approximation=np.pi / 6)
    test_model_on_spherical_harmonics_data_set(
        model_weight_path=f"./models/spherical_harmonics/spherical_harmonics_max_deg_{deg}_norm_1_std_0.25.h5",
        data_set_path=r"C:\git\mainfold_approximation\dataset\coef_extrapolation\spherical_harmonics\test_data_noise_" +
                      f"5_snr_35_norm_regular.pickle",
        model_name="spherical_harmonics_max_deg_9_norm_1_std_0.25",
        deg=5,
        max_deg=9,
        extrapolation_distance_from_approximation=np.pi / 6)

    print("*" * 80)
    print("9")
    print("*" * 80)
    test_ls_on_spherical_harmonics_data_set(
        data_set_path=r"C:\git\mainfold_approximation\dataset\coef_extrapolation\spherical_harmonics\test_data_noise_" +
                      f"9_snr_35_norm_regular.pickle",
        deg_to_predict=deg,
        deg_data_set=9,
        name=f"ls_deg_9",
        extrapolation_distance_from_approximation=np.pi / 6)
    print("*" * 80)
    print("5")
    print("*" * 80)
    test_ls_on_spherical_harmonics_data_set(
        data_set_path=r"C:\git\mainfold_approximation\dataset\coef_extrapolation\spherical_harmonics\test_data_noise_" +
                      f"5_snr_35_norm_regular.pickle",
        deg_to_predict=deg,
        deg_data_set=5,
        name=f"ls_deg_5",
        extrapolation_distance_from_approximation=np.pi / 6)
    print("*" * 80)
    print("3")
    print("*" * 80)
    test_ls_on_spherical_harmonics_data_set(
        data_set_path=r"C:\git\mainfold_approximation\dataset\coef_extrapolation\spherical_harmonics\test_data_noise_" +
                      f"3_snr_35_norm_regular.pickle",
        deg_to_predict=deg,
        deg_data_set=3,
        name=f"ls_deg_3",
        extrapolation_distance_from_approximation=np.pi / 6)

    function_index = 65
    deg_vis = 9

    print("IE")
    error_min_ie, error_max_ie = test_model_on_spherical_harmonics_data_set_on_one_function(
        model_weight_path=f"./models/spherical_harmonics/spherical_harmonics_max_deg_9_norm_1_std_0.25.h5",
        data_set_path=r"C:\git\mainfold_approximation\dataset\coef_extrapolation\spherical_harmonics\test_data_noise_" +
                      f"{deg_vis}_snr_35_norm_regular.pickle",
        max_deg=9,
        function_index=function_index,
        extrapolation_distance_from_approximation=np.pi / 6,
        not_error=False
    )
    print("LS")
    error_min_ls, error_max_ls = test_ls_on_spherical_harmonics_data_set_on_one_function(
        data_set_path=r"C:\git\mainfold_approximation\dataset\coef_extrapolation\spherical_harmonics\test_data_noise_" +
                      f"{deg_vis}_snr_35_norm_regular.pickle",
        deg_to_predict=9,
        function_index=function_index,
        extrapolation_distance_from_approximation=np.pi / 6,
        not_error=False
    )

    error_max = error_max_ls if error_max_ls > error_max_ie else error_max_ie
    error_min = error_min_ls if error_min_ls < error_min_ie else error_min_ie
    print("LS")
    _ = test_ls_on_spherical_harmonics_data_set_on_one_function(
        data_set_path=r"C:\git\mainfold_approximation\dataset\coef_extrapolation\spherical_harmonics\test_data_noise_" +
                      f"{deg_vis}_snr_35_norm_regular.pickle",
        deg_to_predict=9,
        function_index=function_index,
        extrapolation_distance_from_approximation=np.pi / 6,
        not_error=False,
        error_min=error_min,
        error_max=error_max
    )
    print("IE")
    _ = test_model_on_spherical_harmonics_data_set_on_one_function(
        model_weight_path=f"./models/spherical_harmonics/spherical_harmonics_max_deg_9_norm_1_std_0.25.h5",
        data_set_path=r"C:\git\mainfold_approximation\dataset\coef_extrapolation\spherical_harmonics\test_data_noise_" +
                      f"{deg_vis}_snr_35_norm_regular.pickle",
        max_deg=9,
        function_index=function_index,
        extrapolation_distance_from_approximation=np.pi / 6,
        error_min=error_min,
        error_max=error_max,
        not_error=False
    )

    print("True")
    value_min_tr, value_max_tr = plot_spherical_harmonics_data_set_on_one_function(
        data_set_path=r"C:\git\mainfold_approximation\dataset\coef_extrapolation\spherical_harmonics\test_data_noise_" +
                      f"{deg_vis}_snr_35_norm_regular.pickle",
        max_deg=9,
        function_index=function_index,
        extrapolation_distance_from_approximation=np.pi / 6,
    )
    print("LS")
    value_min_ls, value_max_ls = test_ls_on_spherical_harmonics_data_set_on_one_function(
        data_set_path=r"C:\git\mainfold_approximation\dataset\coef_extrapolation\spherical_harmonics\test_data_noise_" +
                      f"{deg_vis}_snr_35_norm_regular.pickle",
        deg_to_predict=9,
        function_index=function_index,
        extrapolation_distance_from_approximation=np.pi / 6,
        not_error=True,
    )
    print("IE")
    value_min_ie, value_max_ie = test_model_on_spherical_harmonics_data_set_on_one_function(
        model_weight_path=f"./models/spherical_harmonics/spherical_harmonics_max_deg_9_norm_1_std_0.25.h5",
        data_set_path=r"C:\git\mainfold_approximation\dataset\coef_extrapolation\spherical_harmonics\test_data_noise_" +
                      f"{deg_vis}_snr_35_norm_regular.pickle",
        max_deg=9,
        function_index=function_index,
        extrapolation_distance_from_approximation=np.pi / 6,
        not_error=True,
    )

    value_min = value_min_tr if value_min_tr < value_min_ls else value_min_ls
    value_min = value_min if value_min < value_min_ie else value_min_ie
    value_max = value_max_tr if value_max_tr > value_max_ls else value_max_ls
    value_max = value_max if value_max > value_max_ie else value_max_ie

    _ = plot_spherical_harmonics_data_set_on_one_function(
        data_set_path=r"C:\git\mainfold_approximation\dataset\coef_extrapolation\spherical_harmonics\test_data_noise_" +
                      f"{deg_vis}_snr_35_norm_regular.pickle",
        max_deg=9,
        function_index=function_index,
        extrapolation_distance_from_approximation=np.pi / 6,
        error_min=value_min,
        error_max=value_max
    )
    print("LS")
    _ = test_ls_on_spherical_harmonics_data_set_on_one_function(
        data_set_path=r"C:\git\mainfold_approximation\dataset\coef_extrapolation\spherical_harmonics\test_data_noise_" +
                      f"{deg_vis}_snr_35_norm_regular.pickle",
        deg_to_predict=9,
        function_index=function_index,
        extrapolation_distance_from_approximation=np.pi / 6,
        not_error=True,
        error_min=value_min,
        error_max=value_max
    )
    print("IE")
    _ = test_model_on_spherical_harmonics_data_set_on_one_function(
        model_weight_path=f"./models/spherical_harmonics/spherical_harmonics_max_deg_9_norm_1_std_0.25.h5",
        data_set_path=r"C:\git\mainfold_approximation\dataset\coef_extrapolation\spherical_harmonics\test_data_noise_" +
                      f"{deg_vis}_snr_35_norm_regular.pickle",
        max_deg=9,
        function_index=function_index,
        extrapolation_distance_from_approximation=np.pi / 6,
        not_error=True,
        error_min=value_min,
        error_max=value_max
    )


if "__main__" == __name__:
    main()
