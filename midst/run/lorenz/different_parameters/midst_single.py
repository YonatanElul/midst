import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from datetime import datetime
from DynamicalSystems import DATA_DIR, LOGS_DIR
from DynamicalSystems.utils.loggers import Logger
from DynamicalSystems.utils.optim import Optimizer
from DynamicalSystems.losses.losses import ModuleLoss, MASELoss
from DynamicalSystems.concurrent_dynamics.models.models import MIDST
from DynamicalSystems.concurrent_dynamics.data.datasets import StrangeAttractorsDataset
from DynamicalSystems.concurrent_dynamics.utils.trainers import ConcurrentDynamicsTrainer

import torch
import numpy as np

# Define global parameters
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# N sub-systems
n_seeds = 5
n_systems = (1, 2, 4, 8, 16, 32)
prediction_steps = 1
residual_dynamics = True
separate_dynamics = False
dmf_dynamics = False
attractor = 'lorenz'


def generate_params():
    # Model parameters
    m_dynamics = 1
    k = 64
    n = 3
    n_encoder_layers = 3
    l0_units = 64
    units_factor = 2
    activation = 'leakyrelu'
    final_activation = None
    norm = None
    dropout = None
    bias = None
    spectral_norm = False
    single_system = True
    model_params = {
        'm_dynamics': m_dynamics,
        'observable_dim': k,
        'states_dim': n,
        'n_encoder_layers': n_encoder_layers,
        'l0_units': l0_units,
        'units_factor': units_factor,
        'activation': activation,
        'final_activation': final_activation,
        'norm': norm,
        'dropout': dropout,
        'bias': bias,
        'k_forward_prediction': prediction_steps,
        'spectral_norm': spectral_norm,
        'single_system': single_system,
    }

    # Optimizer parameters
    optimizer_hparams = {
        'lr': 0.001,
        'weight_decay': 1e-6,
    }

    scheduler_hparams = {
        'mode': 'min',
        'factor': 0.1,
        'patience': 10,
        'threshold': 1e-4,
        'threshold_mode': 'rel',
        'cooldown': 0,
        'min_lr': 1e-6,
        'eps': 1e-8,
        'verbose': True,
    }

    return model_params, optimizer_hparams, scheduler_hparams


model_type = MIDST
logs_dir_ = LOGS_DIR
os.makedirs(logs_dir_, exist_ok=True)
data_dir = os.path.join(DATA_DIR, 'Attractors')
if __name__ == '__main__':
    for s in range(n_seeds):
        for m in n_systems:
            # Paths
            attractors_dirs_names = [
                f'{attractor}_1_attractors_{s}S_{p}P'
                for p in range(m)
            ]
            model_params, optimizer_hparams, scheduler_hparams = generate_params()

            # Data parameters
            data_dirs = [
                os.path.join(data_dir, d)
                for d in attractors_dirs_names
            ]
            temporal_horizon = 64
            overlap = 63
            train_time = 100
            val_time = 10
            test_time = 20
            dt = 0.01
            train_total_trajectory_length = int(train_time / dt)
            val_total_trajectory_length = int(val_time / dt)
            test_total_trajectory_length = int(test_time / dt)

            # Data loader parameters
            batch_size = 16
            num_workers = 8
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pin_memory = True
            drop_last = False

            # Trainer parameters
            num_epochs = 100
            max_elements_to_save = 10
            checkpoints = True
            early_stopping = None
            checkpoints_mode = 'min'

            # Set up the log dir
            date = str(datetime.today()).split()[0]
            k = model_params['observable_dim']
            experiment_name = f"MIDST_SingleDynamics_{attractor}_{m}A_{s}S" \
                              f"{k}K_" \
                              f"{prediction_steps}H_" \
                              f"{'Res_' if residual_dynamics else ''}" \
                              f"{'Sep_' if separate_dynamics else ''}" \
                              f"{'DMF_' if separate_dynamics else ''}" \
                              f"{date}"
            logs_dir = os.path.join(logs_dir_, experiment_name)
            os.makedirs(logs_dir, exist_ok=True)

            # Define the Datasets & Data loaders
            train_data_path = [
                os.path.join(d, 'train_ds.h5')
                for d in data_dirs
            ]
            train_ds = StrangeAttractorsDataset(
                filepath=train_data_path,
                temporal_horizon=temporal_horizon,
                overlap=overlap,
                prediction_horizon=prediction_steps,
                n_systems=m,
            )
            val_data_path = [
                os.path.join(d, 'val_ds.h5')
                for d in data_dirs
            ]
            val_ds = StrangeAttractorsDataset(
                filepath=val_data_path,
                temporal_horizon=temporal_horizon,
                overlap=overlap,
                prediction_horizon=prediction_steps,
                n_systems=m,
            )
            test_data_path = [
                os.path.join(d, 'test_ds.h5')
                for d in data_dirs
            ]
            test_ds = StrangeAttractorsDataset(
                filepath=test_data_path,
                temporal_horizon=temporal_horizon,
                overlap=overlap,
                prediction_horizon=prediction_steps,
                n_systems=m,
            )
            train_dl = torch.utils.data.DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last,
            )
            val_dl = torch.utils.data.DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last,
            )
            test_dl = torch.utils.data.DataLoader(
                test_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last,
            )

            # Set up the model
            model = model_type(
                **model_params
            )

            # Set up the optimizer
            optimizers = [
                torch.optim.AdamW(
                    params=model.parameters(),
                    **optimizer_hparams,
                ),
            ]
            schedulers = [
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer=optimizers[0],
                    **scheduler_hparams
                ),
            ]
            optimizer = Optimizer(
                optimizers=optimizers,
                schedulers=schedulers,
            )

            # Set up the loss and evaluation metrics
            loss_fn = ModuleLoss(
                model=torch.nn.MSELoss()
            )
            evaluation_metric = MASELoss(
                m=prediction_steps,
                trajectory_length=temporal_horizon,
            )

            # Set up the logger
            logger = Logger(
                log_dir=logs_dir_,
                experiment_name=experiment_name,
                max_elements=max_elements_to_save,
            )

            # Write Scenario Specs
            specs = {
                f'{"-" * 10} Data Specs {"-" * 10}': '',
                'm': m,
                'temporal_horizon': temporal_horizon,
                'overlap': overlap,
                f'\n{"-" * 10} DataLoader Specs {"-" * 10}': '',
                'batch_size': batch_size,
                'num_workers': num_workers,
                'pin_memory': pin_memory,
                'drop_last': drop_last,
                f'\n{"-" * 10} Model Specs {"-" * 10}': '',
                'Model': type(model).__name__,
            }
            specs.update(model_params)
            loss_params = {
                f'\n{"-" * 10} Loss Specs {"-" * 10}': '',
                'loss_fn': type(loss_fn.model).__name__,
                'eval_fn': type(evaluation_metric).__name__,
                f'\n{"-" * 10} Trainer Specs {"-" * 10}': '',
                'num_epochs': num_epochs,
                'checkpoints': checkpoints,
                'early_stopping': early_stopping,
                'checkpoints_mode': checkpoints_mode,
            }
            specs.update(loss_params)
            specs[f'\n{"-" * 10} Optimizer Specs {"-" * 10}'] = ''
            specs.update(optimizer_hparams)
            specs.update(scheduler_hparams)

            specs_file = os.path.join(logs_dir, 'experiment_specs.txt')
            with open(specs_file, 'w') as f:
                for k, v in specs.items():
                    f.write(f"{k}: {str(v)}\n")

            # Set up the trainer
            trainer = ConcurrentDynamicsTrainer(
                model=model,
                loss_fn=loss_fn,
                evaluation_metric=evaluation_metric,
                optimizer=optimizer,
                device=device,
                logger=logger,
            )

            # Train the model
            trainer.fit(
                dl_train=train_dl,
                dl_val=val_dl,
                num_epochs=num_epochs,
                checkpoints=checkpoints,
                checkpoints_mode=checkpoints_mode,
                early_stopping=early_stopping,
            )

            # Load the best model
            best_model_path = os.path.join(
                logs_dir,
                "BestModel.PyTorchModule",
            )
            ckpt = torch.load(best_model_path)['model']
            model = model_type(
                **model_params
            )
            model.load_state_dict(ckpt)
            model = model.to(device)

            # Insert the optimal model and new logger to the trainer
            trainer._model = model

            # Test the model
            trainer.evaluate(
                dl_test=test_dl,
                ignore_cap=True,
            )
