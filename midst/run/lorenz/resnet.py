import os

from midst.models.resnet_models import Bottleneck, ResNet2D

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from datetime import datetime
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
date = str(datetime.today()).split()[0]

# N sub-systems
n_seeds = 5
n_systems = (1, 2, 4, 6, 8)
prediction_steps = 1

attractor = 'lorenz'
logs_dir_ = LOGS_DIR

# Data parameters
temporal_horizon = 64
overlap = 63
train_time = 100
val_time = 10
test_time = 20
dt = 0.01
train_total_trajectory_length = int(train_time / dt)
val_total_trajectory_length = int(val_time / dt)
test_total_trajectory_length = int(test_time / dt)
n = 3

# Data loader parameters
batch_size = 16
num_workers = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pin_memory = True
drop_last = False

# Trainer parameters
num_epochs = 100
max_elements_to_save = 2
checkpoints = True
early_stopping = None
checkpoints_mode = 'min'


def generate_params():
    # Model parameters
    n = 3
    input_dim = m
    output_dim = m * n * prediction_steps * temporal_horizon
    block = Bottleneck
    layers = [3, 4, 6, 3]
    replace_stride_with_dilation = (True, True, True)
    activation = 'relu'
    model_params = {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'block': block,
        'layers': layers,
        'replace_stride_with_dilation': replace_stride_with_dilation,
        'prediction_horizon': prediction_steps * temporal_horizon,
        'activation': activation,
    }

    # Optimizer parameters
    optimizer_hparams = {
        'lr': 0.001,
        'weight_decay': 1e-6,
    }
    scheduler_hparams = {
        'mode': 'min',
        'factor': 0.1,
        'patience': 20,
        'threshold': 1e-4,
        'threshold_mode': 'rel',
        'cooldown': 0,
        'min_lr': 1e-6,
        'eps': 1e-8,
        'verbose': True,
    }

    return model_params, optimizer_hparams, scheduler_hparams


noise = None
# noise = {
#     'loc': 0,
#     'scale': 0.1,
# }
if __name__ == '__main__':
    for s in range(3, n_seeds):
        for m in n_systems:
            input_dim = m
            output_dim = m * n * prediction_steps * temporal_horizon

            attractor_type = f'{attractor}_{m}_attractors_{s}'
            attractor_data_type = f'{attractor}_8_attractors_{s}'
            model_params, optimizer_hparams, scheduler_hparams = generate_params()
            model_params['input_dim'] = input_dim
            model_params['output_dim'] = output_dim

            # Set up the log dir
            experiment_name = f"ResNet_{attractor_type}_{m}A_" \
                              f"{prediction_steps}H_{'Noisy_' if noise is not None else ''}{date}"
            logs_dir = os.path.join(logs_dir_, experiment_name)
            os.makedirs(logs_dir, exist_ok=True)
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
                noise=noise,
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
                noise=noise,
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
                noise=noise,
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
            model = ResNet2D(
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
            # agnostic_schedulers = [
            #     torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            #         optimizer=optimizers[0],
            #         **agnostic_scheduler_hparams
            #     ),
            # ]
            optimizer = Optimizer(
                optimizers=optimizers,
                schedulers=schedulers,
                # agnostic_schedulers=agnostic_schedulers
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
                'loss_fn': type(loss_fn).__name__,
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
            model = ResNet2D(
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
