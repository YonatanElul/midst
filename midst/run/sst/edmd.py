from datetime import datetime
from DynamicalSystems import DATA_DIR, LOGS_DIR
from DynamicalSystems.utils.scikit_metrics import ScikitMASE
from DynamicalSystems.concurrent_dynamics.data.datasets import SSTDataset
from DynamicalSystems.models.dmd import EDMD, mpEDMD, PolynomialFeaturesTransform, SVDTransform, PolarTransform
from DynamicalSystems.utils.defaults import GT_TENSOR_INPUTS_KEY, GT_TENSOR_PREDICITONS_KEY

import os
import pickle
import numpy as np

# Define global parameters
seed = 42
np.random.seed(seed)
date = str(datetime.today()).split()[0]

# Setup
single = True
use_transformation = True
eigvals_cutoff_threshold = 1e-2
transformation_type = 'svd'
degree = 4
if use_transformation:
    if transformation_type == 'poly':
        transform = PolynomialFeaturesTransform(degree=degree)

    elif transformation_type == 'svd':
        transform = SVDTransform()

    elif transformation_type == 'polar':
        transform = PolarTransform()

model_name = 'EDMD'
model_type = EDMD
model_params = {
    'eigvals_cutoff_threshold': eigvals_cutoff_threshold,
}
experiment_name = (
    f"{model_name}_SST_"
    f"{'SingleKO_' if single else ''}"
    f"{f'{transformation_type}_transformed_' if use_transformation else ''}"
    f"{date}"
)
logs_dir = os.path.join(LOGS_DIR, experiment_name)
os.makedirs(logs_dir, exist_ok=True)
if __name__ == '__main__':
    datasets_paths = (
        os.path.join(DATA_DIR, "SSTV2", "sst.wkmean.1990-present.nc"),
    )
    mask_path = os.path.join(DATA_DIR, "SSTV2", "lsmask.nc")
    prediction_horizon = 1
    trajectory_length = 16
    val_ratio = 0.1
    test_ratio = 0.2
    n_systems = 50
    train_ds = SSTDataset(
        mode='Train',
        datasets_paths=datasets_paths,
        surface_mask_path=mask_path,
        temporal_horizon=trajectory_length,
        prediction_horizon=prediction_horizon,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        simple_prediction=True,
        top_lat=(40, 90),
        bottom_lat=(60, 110),
        left_long=(190, 70),
        right_long=(210, 90),
    )
    val_ds = SSTDataset(
        mode='Val',
        datasets_paths=datasets_paths,
        surface_mask_path=mask_path,
        temporal_horizon=trajectory_length,
        prediction_horizon=prediction_horizon,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        simple_prediction=True,
        top_lat=(40, 90),
        bottom_lat=(60, 110),
        left_long=(190, 70),
        right_long=(210, 90),
    )
    test_ds = SSTDataset(
        mode='Test',
        datasets_paths=datasets_paths,
        surface_mask_path=mask_path,
        temporal_horizon=trajectory_length,
        prediction_horizon=prediction_horizon,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        simple_prediction=True,
        top_lat=(40, 90),
        bottom_lat=(60, 110),
        left_long=(190, 70),
        right_long=(210, 90),
    )
    evaluation_metric = ScikitMASE(
        trajectory_length=trajectory_length,
        m=prediction_horizon,
    )

    # Gather the data
    train_samples_x = [train_ds[0][GT_TENSOR_INPUTS_KEY].numpy()[:, [-1], :]] + [
        train_ds[i][GT_TENSOR_INPUTS_KEY].numpy()[:, [-1], :]
        for i in range(1, len(train_ds))
    ]
    train_samples_y = [train_ds[0][GT_TENSOR_PREDICITONS_KEY].numpy()] + [
        train_ds[i][GT_TENSOR_PREDICITONS_KEY].numpy()[:, [-1], :]
        for i in range(1, len(train_ds))
    ]
    train_samples_x = np.concatenate(train_samples_x, axis=1)
    train_samples_x = np.swapaxes(train_samples_x, axis1=0, axis2=1)
    train_samples_y = np.concatenate(train_samples_y, axis=1)
    train_samples_y = np.swapaxes(train_samples_y, axis1=0, axis2=1)

    val_samples_x = [val_ds[0][GT_TENSOR_INPUTS_KEY].numpy()[:, [-1], :]] + [
        val_ds[i][GT_TENSOR_INPUTS_KEY].numpy()[:, [-1], :]
        for i in range(1, len(val_ds))
    ]
    val_samples_y = [val_ds[0][GT_TENSOR_PREDICITONS_KEY].numpy()] + [
        val_ds[i][GT_TENSOR_PREDICITONS_KEY].numpy()[:, [-1], :]
        for i in range(1, len(val_ds))
    ]
    val_samples_x = np.concatenate(val_samples_x, axis=1)
    val_samples_x = np.swapaxes(val_samples_x, axis1=0, axis2=1)
    val_samples_y = np.concatenate(val_samples_y, axis=1)
    val_samples_y = np.swapaxes(val_samples_y, axis1=0, axis2=1)

    train_samples_x = np.concatenate(
        [train_samples_x, val_samples_x],
        axis=0,
    )
    train_samples_y = np.concatenate(
        [train_samples_y, val_samples_y],
        axis=0,
    )

    test_samples_x = [test_ds[0][GT_TENSOR_INPUTS_KEY].numpy()[:, [-1], :]] + [
        test_ds[i][GT_TENSOR_INPUTS_KEY].numpy()[:, [-1], :]
        for i in range(1, len(test_ds))
    ]
    test_samples_y = [test_ds[0][GT_TENSOR_PREDICITONS_KEY].numpy()] + [
        test_ds[i][GT_TENSOR_PREDICITONS_KEY].numpy()[:, [-1], :]
        for i in range(1, len(test_ds))
    ]
    test_samples_x = np.concatenate(test_samples_x, axis=1)
    test_samples_x = np.swapaxes(test_samples_x, axis1=0, axis2=1)
    test_samples_y = np.concatenate(test_samples_y, axis=1)
    test_samples_y = np.swapaxes(test_samples_y, axis1=0, axis2=1)

    if single:
        if transformation_type == 'poly' and use_transformation:
            # Apply polynomial transformation to each region separately
            n_regions = train_samples_x.shape[1]
            koopman_operators = []
            eval_score_train = []
            eval_score_test = []
            predictions = []
            train_scores_mase_pointwise = []
            test_scores_mase_pointwise = []
            for i in range(n_regions):
                print(f"Region {i + 1} / {n_regions}")
                x_train = train_samples_x[:, i, :]
                y_train = train_samples_y[:, i, :]
                x_test = test_samples_x[:, i, :]
                y_test = test_samples_y[:, i, :]

                # Train the models
                model = model_type(
                    **model_params
                )
                if use_transformation:
                    if transformation_type == 'polar':
                        n_samples = x_train.shape[0] + x_test.shape[0]
                        train_times = np.arange(
                            start=0,
                            stop=x_train.shape[0],
                        ) / n_samples
                        test_times = np.arange(
                            start=x_train.shape[0],
                            stop=x_train.shape[0] + x_test.shape[0],
                        ) / n_samples
                        embedded_x_train = transform.transform(X=x_train, times=train_times)
                        embedded_x_test = transform.transform(X=x_test, times=test_times)

                    else:
                        embedded_x_train = transform.transform(X=x_train)
                        embedded_x_test = transform.transform(X=x_test)

                else:
                    embedded_x_train = x_train
                    embedded_x_test = x_test

                model.fit(embedded_x_train)
                y_pred_train = model.predict(embedded_x_train)
                y_pred_test = model.predict(embedded_x_test)

                if use_transformation:
                    y_pred_train = transform.inverse_transform(
                        initial_condition=x_train[[0]],
                        embedding=y_pred_train,
                    )
                    y_pred_test = transform.inverse_transform(
                        initial_condition=x_test[[0]],
                        embedding=y_pred_test,
                    )

                train_score = evaluation_metric(
                    x=x_train[None, None, ...],
                    y=y_train[None, None, ...],
                    y_pred=y_pred_train[None, None, ...],
                )
                test_score = evaluation_metric(
                    x=x_test[None, None, ...],
                    y=y_test[None, None, ...],
                    y_pred=y_pred_test[None, None, ...],
                )
                preds = {
                    'x_gt': test_samples_x,
                    'y': test_samples_y,
                    'y_pred': y_pred_test,
                }

                print(f"Test Score: {test_score}")

                naive_train = np.diff(y_train, axis=0).mean(axis=1)
                naive_test = np.diff(y_test, axis=0).mean(axis=1)
                train_mase_pointwise = np.abs(y_pred_train - y_train).mean(axis=1)
                test_mase_pointwise = np.abs(y_pred_test - y_test).mean(axis=1)

                koopman_operators.append(model.koopman)
                eval_score_train.append(train_score)
                eval_score_test.append(test_score)
                predictions.append(preds)
                train_scores_mase_pointwise.append(train_mase_pointwise)
                test_scores_mase_pointwise.append(test_mase_pointwise)

            with open(os.path.join(logs_dir, 'train_scores.pkl'), 'wb') as f:
                pickle.dump(
                    file=f,
                    obj=train_score,
                )
            with open(os.path.join(logs_dir, 'test_scores.pkl'), 'wb') as f:
                pickle.dump(
                    file=f,
                    obj=test_score,
                )
            with open(os.path.join(logs_dir, 'predictions.pkl'), 'wb') as f:
                pickle.dump(
                    file=f,
                    obj=predictions,
                )
            with open(os.path.join(logs_dir, 'koopman_operator.pkl'), 'wb') as f:
                pickle.dump(
                    file=f,
                    obj=koopman_operators,
                )
            with open(os.path.join(logs_dir, 'train_scores_mase_pointwise.pkl'), 'wb') as f:
                pickle.dump(
                    file=f,
                    obj=train_scores_mase_pointwise,
                )
            with open(os.path.join(logs_dir, 'test_scores_mase_pointwise.pkl'), 'wb') as f:
                pickle.dump(
                    file=f,
                    obj=test_scores_mase_pointwise,
                )

        else:
            # Reshape into a 2D matrix
            train_samples_x = np.reshape(train_samples_x, (train_samples_x.shape[0], -1))
            train_samples_y = np.reshape(train_samples_y, (train_samples_y.shape[0], -1))
            test_samples_x = np.reshape(test_samples_x, (test_samples_x.shape[0], -1))
            test_samples_y = np.reshape(test_samples_y, (test_samples_y.shape[0], -1))

            # Train the models
            model = model_type(
                **model_params
            )
            if use_transformation:
                if transformation_type == 'polar':
                    n_samples = train_samples_x.shape[0] + test_samples_x.shape[0]
                    train_times = np.arange(
                        start=0,
                        stop=train_samples_x.shape[0],
                    ) / n_samples
                    test_times = np.arange(
                        start=train_samples_x.shape[0],
                        stop=train_samples_x.shape[0] + test_samples_x.shape[0],
                    ) / n_samples
                    embedded_x_train = transform.transform(X=train_samples_x, times=train_times)
                    embedded_x_test = transform.transform(X=test_samples_x, times=test_times)

                else:
                    embedded_x_train = transform.transform(X=train_samples_x)
                    embedded_x_test = transform.transform(X=test_samples_x)

            else:
                embedded_x_train = train_samples_x
                embedded_x_test = test_samples_x

            model.fit(embedded_x_train)
            y_pred_train = model.predict(embedded_x_train)
            y_pred_test = model.predict(embedded_x_test)

            if use_transformation:
                y_pred_train = transform.inverse_transform(
                    initial_condition=train_samples_x[[0]],
                    embedding=y_pred_train,
                )
                y_pred_test = transform.inverse_transform(
                    initial_condition=test_samples_x[[0]],
                    embedding=y_pred_test,
                )

            train_score = evaluation_metric(
                x=train_samples_x[None, None, ...],
                y=train_samples_y[None, None, ...],
                y_pred=y_pred_train[None, None, ...],
            )
            test_score = evaluation_metric(
                x=test_samples_x[None, None, ...],
                y=test_samples_y[None, None, ...],
                y_pred=y_pred_test[None, None, ...],
            )

            naive_train = np.diff(train_samples_y, axis=0).mean(axis=1)
            naive_test = np.diff(test_samples_y, axis=0).mean(axis=1)
            train_scores_mase_pointwise = np.abs(y_pred_train - train_samples_y).mean(axis=1)
            test_scores_mase_pointwise = np.abs(y_pred_test - test_samples_y).mean(axis=1)

            predictions = {
                'x_gt': test_samples_x,
                'y': test_samples_y,
                'y_pred': y_pred_test,
            }
            with open(os.path.join(logs_dir, 'train_scores_mase_pointwise.pkl'), 'wb') as f:
                pickle.dump(
                    file=f,
                    obj=train_scores_mase_pointwise,
                )
            with open(os.path.join(logs_dir, 'test_scores_mase_pointwise.pkl'), 'wb') as f:
                pickle.dump(
                    file=f,
                    obj=test_scores_mase_pointwise,
                )
            with open(os.path.join(logs_dir, 'train_scores.pkl'), 'wb') as f:
                pickle.dump(
                    file=f,
                    obj=train_score,
                )
            with open(os.path.join(logs_dir, 'test_scores.pkl'), 'wb') as f:
                pickle.dump(
                    file=f,
                    obj=test_score,
                )
            with open(os.path.join(logs_dir, 'predictions.pkl'), 'wb') as f:
                pickle.dump(
                    file=f,
                    obj=predictions,
                )
            with open(os.path.join(logs_dir, 'koopman_operator.pkl'), 'wb') as f:
                pickle.dump(
                    file=f,
                    obj=model.koopman,
                )

            print(f"Mean Test Score: {np.mean(test_scores_mase_pointwise)}")

    else:
        train_scores = []
        test_scores = []
        for m in range(n_systems):
            system_train_samples_x = train_samples_x[:, [m]]
            system_test_samples_x = test_samples_x[:, [m]]
            system_train_samples_y = train_samples_y[:, [m]]
            system_test_samples_y = test_samples_y[:, [m]]

            if use_transformation:
                embedded_x_train = transform.transform(X=system_train_samples_x)
                embedded_x_test = transform.transform(X=system_test_samples_x)

            else:
                embedded_x_train = system_train_samples_x
                embedded_x_test = system_test_samples_x

            model.fit(embedded_x_train)

            y_pred_train = model.predict(embedded_x_train)
            y_pred_test = model.predict(embedded_x_test)

            if use_transformation:
                y_pred_train = transform.inverse_transform(
                    initial_condition=system_train_samples_x[[0]],
                    embedding=y_pred_train,
                )
                y_pred_test = transform.inverse_transform(
                    initial_condition=system_test_samples_x[[0]],
                    embedding=y_pred_test,
                )

                if use_time_delay_embedding:
                    system_train_samples_x = system_train_samples_x[1:]
                    system_train_samples_y = system_train_samples_y[1:]
                    system_test_samples_y = system_test_samples_y[1:]
                    y_pred_train = y_pred_train[1:]
                    y_pred_test = y_pred_test[1:]

            train_score = evaluation_metric(
                x=system_train_samples_x[None, None, ...],
                y=system_train_samples_y[None, None, ...],
                y_pred=y_pred_train[None, None, ...],
            )
            test_score = evaluation_metric(
                x=system_test_samples_x[None, None, ...],
                y=system_test_samples_y[None, None, ...],
                y_pred=y_pred_test[None, None, ...],
            )

            train_scores.append(train_score)
            test_scores.append(test_score)

        with open(os.path.join(logs_dir, 'train_scores.pkl'), 'wb') as f:
            pickle.dump(
                file=f,
                obj=train_scores,
            )
        with open(os.path.join(logs_dir, 'test_scores.pkl'), 'wb') as f:
            pickle.dump(
                file=f,
                obj=test_scores,
            )
