from datetime import datetime
from DynamicalSystems import DATA_DIR, LOGS_DIR
from DynamicalSystems.utils.scikit_metrics import ScikitMASE
from DynamicalSystems.models.dmd import EDMD, mpEDMD, TimeDelayTransform
from DynamicalSystems.concurrent_dynamics.data.datasets import StrangeAttractorsDataset
from DynamicalSystems.utils.defaults import GT_TENSOR_INPUTS_KEY, GT_TENSOR_PREDICITONS_KEY

import os
import pickle
import numpy as np

# Define global parameters
seed = 42
np.random.seed(seed)
date = str(datetime.today()).split()[0]

# N sub-systems
n_seeds = 5
n_systems = (1, 2, 4, 6, 8)
prediction_steps = 1

# Paths
logs_dir = LOGS_DIR
attractor = 'lorenz'
single = True
use_time_delay_embedding = True
eigvals_cutoff_threshold = 1e-4
k_delays = 1
transform = TimeDelayTransform(k_delays=k_delays)
model_name = 'EDMD'
model_type = EDMD
model_params = {
    'eigvals_cutoff_threshold': eigvals_cutoff_threshold,
}
experiment_name = f"{model_name}_{attractor}_{'SingleKO_' if single else ''}{'Takens_' if use_time_delay_embedding else ''}{date}"
logs_dir = os.path.join(logs_dir, experiment_name)
os.makedirs(logs_dir, exist_ok=True)

# Data parameters
temporal_horizon = 1
overlap = 0
train_time = 100
val_time = 10
test_time = 20
dt = 0.01
train_total_trajectory_length = int(train_time / dt)
val_total_trajectory_length = int(val_time / dt)
test_total_trajectory_length = int(test_time / dt)
maximum_training_len = None
if __name__ == '__main__':
    koopman_operators = {
        m: []
        for m in n_systems
    }
    eval_score_train = {
        m: []
        for m in n_systems
    }
    eval_score_test = {
        m: []
        for m in n_systems
    }
    if single:
        for m in n_systems:
            for s in range(n_seeds):
                print(f"Single: {m} Systems - seed {s + 1} / {n_seeds}")
                attractor_type = f'{attractor}_{m}_attractors_{s}'
                attractor_data_type = f'{attractor}_8_attractors_{s}'

                # Set up the log dir
                data_dir = os.path.join(DATA_DIR, 'Attractors', attractor_data_type)

                # Define the Datasets & Data loaders
                attractors_path = os.path.join(data_dir, 'attractors.pkl')
                train_data_path = os.path.join(data_dir, 'train_ds.h5')
                train_ds = StrangeAttractorsDataset(
                    filepath=train_data_path,
                    temporal_horizon=temporal_horizon,
                    overlap=overlap,
                    time=train_time,
                    dt=dt,
                    attractors_path=attractors_path,
                    prediction_horizon=prediction_steps,
                    n_systems=m,
                )
                val_data_path = os.path.join(data_dir, 'val_ds.h5')
                val_ds = StrangeAttractorsDataset(
                    filepath=val_data_path,
                    temporal_horizon=temporal_horizon,
                    overlap=overlap,
                    time=train_time,
                    dt=dt,
                    attractors_path=attractors_path,
                    prediction_horizon=prediction_steps,
                    n_systems=m,
                )
                test_data_path = os.path.join(data_dir, 'test_ds.h5')
                test_ds = StrangeAttractorsDataset(
                    filepath=test_data_path,
                    temporal_horizon=temporal_horizon,
                    overlap=overlap,
                    time=train_time,
                    dt=dt,
                    attractors_path=attractors_path,
                    prediction_horizon=prediction_steps,
                    n_systems=m,
                )

                evaluation_metric = ScikitMASE(
                    trajectory_length=temporal_horizon,
                    m=prediction_steps,
                )

                # Gather the data
                train_samples_x = [train_ds[0][GT_TENSOR_INPUTS_KEY].numpy()] + [
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

                val_samples_x = [val_ds[0][GT_TENSOR_INPUTS_KEY].numpy()] + [
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

                test_samples_x = [test_ds[0][GT_TENSOR_INPUTS_KEY].numpy()] + [
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

                # Concatenate all different trajectories
                train_samples_x = np.reshape(train_samples_x, (-1, 3))
                train_samples_y = np.reshape(train_samples_y, (-1, 3))
                test_samples_x = np.reshape(test_samples_x, (-1, 3))
                test_samples_y = np.reshape(test_samples_y, (-1, 3))

                if maximum_training_len is not None:
                    train_samples_x = train_samples_x[-maximum_training_len:]
                    train_samples_y = train_samples_y[-maximum_training_len:]

                # Train the models
                model = model_type(
                    **model_params
                )

                if use_time_delay_embedding:
                    embedded_x_train = transform.transform(X=train_samples_x)
                    embedded_x_test = transform.transform(X=test_samples_x)

                else:
                    embedded_x_train = train_samples_x
                    embedded_x_test = test_samples_x

                model.fit(embedded_x_train)

                y_pred_train = model.predict(embedded_x_train)
                y_pred_test = model.predict(embedded_x_test)

                if use_time_delay_embedding:
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
                eval_score_train[m].append(train_score)
                eval_score_test[m].append(test_score)
                koopman_operators[m].append(model.koopman)

        with open(os.path.join(logs_dir, 'train_scores.pkl'), 'wb') as f:
            pickle.dump(
                file=f,
                obj=eval_score_train,
            )
        with open(os.path.join(logs_dir, 'test_scores.pkl'), 'wb') as f:
            pickle.dump(
                file=f,
                obj=eval_score_test,
            )
        with open(os.path.join(logs_dir, 'koopman_operators.pkl'), 'wb') as f:
            pickle.dump(
                file=f,
                obj=koopman_operators,
            )

    else:
        for s in range(n_seeds):
            for m in n_systems:
                attractor_type = f'{attractor}_{m}_attractors_{s}'
                attractor_data_type = f'{attractor}_8_attractors_{s}'

                # Set up the log dir
                data_dir = os.path.join(DATA_DIR, 'Attractors', attractor_data_type)

                # Define the Datasets & Data loaders
                attractors_path = os.path.join(data_dir, 'attractors.pkl')
                train_data_path = os.path.join(data_dir, 'train_ds.h5')
                train_ds = StrangeAttractorsDataset(
                    filepath=train_data_path,
                    temporal_horizon=temporal_horizon,
                    overlap=overlap,
                    time=train_time,
                    dt=dt,
                    attractors_path=attractors_path,
                    prediction_horizon=prediction_steps,
                    n_systems=m,
                )
                val_data_path = os.path.join(data_dir, 'val_ds.h5')
                val_ds = StrangeAttractorsDataset(
                    filepath=val_data_path,
                    temporal_horizon=temporal_horizon,
                    overlap=overlap,
                    time=train_time,
                    dt=dt,
                    attractors_path=attractors_path,
                    prediction_horizon=prediction_steps,
                    n_systems=m,
                )
                test_data_path = os.path.join(data_dir, 'test_ds.h5')
                test_ds = StrangeAttractorsDataset(
                    filepath=test_data_path,
                    temporal_horizon=temporal_horizon,
                    overlap=overlap,
                    time=train_time,
                    dt=dt,
                    attractors_path=attractors_path,
                    prediction_horizon=prediction_steps,
                    n_systems=m,
                )

                evaluation_metric = ScikitMASE(
                    trajectory_length=temporal_horizon,
                    m=prediction_steps,
                )

                # Gather the data
                train_samples_x = [train_ds[0][GT_TENSOR_INPUTS_KEY].numpy()] + [
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

                val_samples_x = [val_ds[0][GT_TENSOR_INPUTS_KEY].numpy()] + [
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

                test_samples_x = [test_ds[0][GT_TENSOR_INPUTS_KEY].numpy()] + [
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

                # Concatenate all different trajectories
                train_samples_x = np.reshape(train_samples_x, (-1, 3))
                train_samples_y = np.reshape(train_samples_y, (-1, 3))
                test_samples_x = np.reshape(test_samples_x, (-1, 3))
                test_samples_y = np.reshape(test_samples_y, (-1, 3))

                # Train the models
                model = model_type(
                    **model_params
                )

                if use_time_delay_embedding:
                    embedded_x_train = transform.transform(X=train_samples_x)
                    embedded_x_test = transform.transform(X=test_samples_x)

                else:
                    embedded_x_train = train_samples_x
                    embedded_x_test = test_samples_x

                model.fit(embedded_x_train)

                y_pred_train = model.predict(embedded_x_train)
                y_pred_test = model.predict(embedded_x_test)

                if use_time_delay_embedding:
                    y_pred_train = transform.inverse_transform(
                        initial_condition=train_samples_x[[0]],
                        embedding=y_pred_train,
                    )
                    y_pred_test = transform.inverse_transform(
                        initial_condition=test_samples_x[[0]],
                        embedding=y_pred_test,
                    )

                    train_samples_x = train_samples_x[1:]
                    train_samples_y = train_samples_y[1:]
                    test_samples_y = test_samples_y[1:]
                    y_pred_train = y_pred_train[1:]
                    y_pred_test = y_pred_test[1:]

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
                eval_score_train[m].append(train_score)
                eval_score_test[m].append(test_score)

        with open(os.path.join(logs_dir, 'train_scores.pkl'), 'wb') as f:
            pickle.dump(
                file=f,
                obj=eval_score_train,
            )
        with open(os.path.join(logs_dir, 'test_scores.pkl'), 'wb') as f:
            pickle.dump(
                file=f,
                obj=eval_score_test,
            )



