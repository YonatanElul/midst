from midst import DATA_DIR
from midst.data.synthetic_data import Attractors
from midst.data.datasets import StrangeAttractorsDataset

import os
import torch
import pickle
import numpy as np

seeds = (0, 7, 42, 8783, 888888)

# Define the log dir
m_systems = 16
noise_dist = None
attractor_types = ('lorenz', )
logs_dir_ = os.path.join(DATA_DIR, 'LorenzAttractors')
os.makedirs(logs_dir_, exist_ok=True)

if __name__ == '__main__':
    for s, seed in enumerate(seeds):
        print(f'Generating data with seed {seed}')
        # Set seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        for attractor_type in attractor_types:
            experiment_name = (
                f"{'noisy_' if noise_dist is not None else ''}{attractor_type}_{m_systems}_attractors_{s}"
            )
            logs_dir = os.path.join(logs_dir_, experiment_name)
            os.makedirs(logs_dir, exist_ok=True)

            # Generate Dynamics object
            attractor_params = None
            x0 = tuple(
                np.random.normal(loc=0, scale=0.5, size=(3,))
                for _ in range(m_systems)
            )
            attractors = Attractors(
                m_attractors=m_systems,
                attractor_type=attractor_type,
                attractor_params=attractor_params,
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
