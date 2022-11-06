import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from datetime import datetime
from midst.models.midst_model import MIDST
from midst import DATA_DIR, LOGS_DIR
from midst.utils.loggers import Logger
from midst.optim.optim import Optimizer
from midst.losses.losses import ModuleLoss, MASELoss
from midst.data.datasets import StrangeAttractorsDataset
from midst.utils.trainers import InterrelatedDynamicsTrainer

import torch
import numpy as np

# Define global parameters
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Paths
date = str(datetime.today()).split()[0]
m = 1
old_m = 8
prediction_steps = 1
prediction_horizon = 20
residual_dynamics = True
dmf_dynamics = True
separate_dynamics = False
dont_load_encoder_decoder = False
new = False
attractor_type = 'lorenz'
noise = None
temporal_horizon = 64
overlap = 63
train_time = 20
val_time = 10
test_time = 20
dt = 0.01
k = 16

logs_dir = LOGS_DIR
seed = 0
trained_date = '2022-06-01'
trained_model_dir_path = os.path.join(
    logs_dir,
    f"MIDST_{attractor_type}_"
    f"New{'NoED_' if dont_load_encoder_decoder else '_'}"
    f"{old_m}PA_"
    f"{m}A_"
    f"{seed}_"
    f"{k}K_"
    f"{prediction_steps}H_"
    f"{train_time}TT_"
    f"{'Res_' if residual_dynamics else ''}"
    f"{'Sep_' if separate_dynamics else ''}"
    f"{'Noisy_' if noise is not None else ''}"
    f"{trained_date}"
)

# Data loader parameters
batch_size = 16
num_workers = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pin_memory = True
drop_last = False

# Model parameters
n = 3
n_encoder_layers = 3
l0_units = 64
units_factor = 2
activation = 'leakyrelu'
final_activation = None
norm = None
dropout = None
bias = None
min_init_eig_val = None
max_init_eig_val = None
identical_sigmas = False
symmetric = False
spectral_norm = False
temporal_sharing = False if separate_dynamics else True
model_params = {
    'm_dynamics': m,
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
    'min_init_eig_val': min_init_eig_val,
    'max_init_eig_val': max_init_eig_val,
    'k_forward_prediction': prediction_steps,
    'identical_sigmas': identical_sigmas,
    'symmetric': symmetric,
    'residual_dynamics': residual_dynamics,
    'dmf_dynamics': dmf_dynamics,
    'separate_dynamics': separate_dynamics,
    'spectral_norm': spectral_norm,
}
model_type = MIDST

# Optimizer parameters
optimizer_hparams = {
    'lr': 0.001,
    'weight_decay': 1e-6,
}
k_optimizer_hparams = optimizer_hparams
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

# Trainer parameters
num_epochs = 1
max_elements_to_save = 2
checkpoints = True
early_stopping = None
checkpoints_mode = 'min'

# Set up the log dir
experiment_name = f"MIDST_{'NewAE_' if new else ''}{attractor_type}_AR_Predictions_" \
                  f"{m}A_{seed}_16K_{prediction_steps}H_{train_time}TT_{prediction_horizon}FS_" \
                  f"{'Res_' if residual_dynamics else ''}{'Sep_' if separate_dynamics else ''}" \
                  f"{'Noisy_' if noise is not None else ''}{date}"
logs_dir = os.path.join(logs_dir, experiment_name)
os.makedirs(logs_dir, exist_ok=True)

attractor_data_type = f'{attractor_type}_8_attractors_{seed}'
data_dir = os.path.join(DATA_DIR, 'Attractors', attractor_data_type)

# Define the Datasets & Data loaders
attractors_path = os.path.join(data_dir, 'attractors.pkl')
test_data_path = os.path.join(logs_dir, 'test_ds.h5')

test_ds = StrangeAttractorsDataset(
    filepath=test_data_path,
    temporal_horizon=temporal_horizon,
    overlap=overlap,
    time=test_time,
    dt=dt,
    attractors_path=attractors_path,
    prediction_horizon=prediction_steps,
    noise=noise,
    system_ind=[9, ],
)
if __name__ == '__main__':
    for starting_ind in range(0, (len(test_ds) - prediction_horizon), prediction_horizon):
        experiment_name = f"SI_{starting_ind}"
        logs_dir_ = os.path.join(logs_dir, experiment_name)
        os.makedirs(logs_dir_, exist_ok=True)

        # Set up the model
        model_ckpt_path = os.path.join(trained_model_dir_path, 'BestModel.PyTorchModule')
        ckpt = torch.load(model_ckpt_path, map_location=device)['model']
        model = model_type(
            **model_params
        )
        model.load_state_dict(ckpt, strict=False, )
        model.to(device)

        # Set up the optimizer
        optimizers = [
            torch.optim.AdamW(
                params=model.singular_values_params(),
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
            log_dir=logs_dir,
            experiment_name=experiment_name,
            max_elements=max_elements_to_save,
        )

        # Write Scenario Specs
        specs = {
            f'{"-" * 10} Data Specs {"-" * 10}': '',
            'm': m,
            'n': n,
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

        specs_file = os.path.join(logs_dir_, 'experiment_specs.txt')
        with open(specs_file, 'w') as f:
            for k, v in specs.items():
                f.write(f"{k}: {str(v)}\n")

        # Set up the trainer
        trainer = InterrelatedDynamicsTrainer(
            model=model,
            loss_fn=loss_fn,
            evaluation_metric=evaluation_metric,
            optimizer=optimizer,
            device=device,
            logger=logger,
        )

        # Test the model
        trainer.forecast(
            ds_test=test_ds,
            horizon=prediction_horizon,
            starting_ind=starting_ind,
        )
