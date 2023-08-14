from DynamicalSystems import LOGS_DIR, DATA_DIR
from DynamicalSystems.concurrent_dynamics.data.synthetic_data import Attractors
from DynamicalSystems.concurrent_dynamics.data.datasets import StrangeAttractorsDataset

import os
import torch
import pickle
import numpy as np

seeds = (42, 7759, 8783, 18899, 18889919)

# Define the log dir
m_systems = 1
noise_dist = None
attractor_types = ('lorenz',)
params_per_attracotr = (
    (
            [
                {
                    'sigma': sigma,
                    'rho': 28,
                    'beta': 8 / 3,
                }
                for sigma in (7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)
            ] +
            [
                {
                    'sigma': 10,
                    'rho': rho,
                    'beta': 8 / 3,
                }
                for rho in (30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56)
            ] +
            [
                {
                    'sigma': 10,
                    'rho': 28,
                    'beta': beta,
                }
                for beta in (4 / 3, 5 / 3, 6 / 3, 7 / 3)
            ]
    ),
)
logs_dir_ = os.path.join(DATA_DIR, 'Attractors')
os.makedirs(logs_dir_, exist_ok=True)

if __name__ == '__main__':
    for s, seed in enumerate(seeds):
        print(f'Generating data with seed {seed}')
        # Set seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        for a_i, attractor_type in enumerate(attractor_types):
            for p_i, params in enumerate(params_per_attracotr[a_i]):
                experiment_name = (
                    f"{'noisy_' if noise_dist is not None else ''}{attractor_type}_{m_systems}_attractors_{s}S_{p_i}P"
                )
                logs_dir = os.path.join(logs_dir_, experiment_name)
                os.makedirs(logs_dir, exist_ok=True)

                # Generate Dynamics object
                x0 = tuple(
                    np.concatenate(
                        [
                            np.random.normal(loc=0, scale=0.1, size=(1,)),
                            np.random.normal(loc=0, scale=0.1, size=(1,)),
                            np.array([0, ]),
                        ]
                    )
                    for _ in range(m_systems)
                )
                attractors = Attractors(
                    m_attractors=m_systems,
                    attractor_type=attractor_type,
                    attractor_params=params,
                    x0=x0,
                    noise_dist=noise_dist,
                )

                # Log the training dynamics object
                with open(os.path.join(logs_dir, "attractors.pkl"), 'wb') as f:
                    pickle.dump(obj=attractors, file=f)

                # Define the Datasets & Data loaders
                temporal_horizon = 64
                overlap = 63
                train_time = 500
                val_time = 50
                test_time = 100
                dt = 0.01
                prediction_horizon = 1

                train_ds = StrangeAttractorsDataset(
                    temporal_horizon=temporal_horizon,
                    overlap=overlap,
                    time=train_time,
                    dt=dt,
                    attractors=attractors,
                    filepath=os.path.join(logs_dir, 'train_ds.h5'),
                    prediction_horizon=prediction_horizon,
                )

                # Advance the validation set 1 temporal_horizon samples into the future, to avoid overlap with training set
                attractors.compute_trajectory(n=temporal_horizon, dt=dt)
                val_ds = StrangeAttractorsDataset(
                    temporal_horizon=temporal_horizon,
                    overlap=overlap,
                    time=val_time,
                    dt=dt,
                    attractors=attractors,
                    filepath=os.path.join(logs_dir, 'val_ds.h5'),
                    prediction_horizon=prediction_horizon,
                )

                # Advance the test set 1 temporal_horizon samples into the future, to avoid overlap with validation set
                attractors.compute_trajectory(n=temporal_horizon, dt=dt)
                test_ds = StrangeAttractorsDataset(
                    temporal_horizon=temporal_horizon,
                    overlap=overlap,
                    time=test_time,
                    dt=dt,
                    attractors=attractors,
                    filepath=os.path.join(logs_dir, 'test_ds.h5'),
                    prediction_horizon=prediction_horizon,
                )
