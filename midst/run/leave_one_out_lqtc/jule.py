import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from datetime import datetime
from midst import DATA_DIR, LOGS_DIR
from midst.utils.loggers import Logger
from midst.optim.optim import Optimizer
from midst.utils.trainers import LQTSTrainer
from midst.losses.losses import MASELoss, UniversalLinearEmbeddingLoss
from midst.data.datasets import ECGRDVQLeaveOneOutDataLoadersGenerator
from midst.models.universal_linear_embeddings_ae import UniversalLinearEmbeddingAE

import torch
import numpy as np

if __name__ == '__main__':
    # Set seed
    seed = 8783
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the log dir
    observable_dim = 64
    n_encoder_layers = 4
    l0_units = 64
    trajectory_length = 6
    prediction_horizon = 10
    date = str(datetime.today()).split()[0]
    experiment_name = f"LQTC_Leave1Out_JULE_{observable_dim}K_{date}"
    logs_dir = os.path.join(LOGS_DIR, experiment_name)
    os.makedirs(logs_dir, exist_ok=True)

    # Define the Datasets & Data loaders
    data_dir = os.path.join(DATA_DIR, "ECGRDVQ")
    train_dir = os.path.join(data_dir, 'Train')
    val_dir = os.path.join(data_dir, 'Val')
    test_dir = os.path.join(data_dir, 'Test')
    datasets_save_dir = os.path.join(data_dir, 'Leave1Out')
    os.makedirs(datasets_save_dir, exist_ok=True)
    n_val_patients = 4
    batch_size = 16
    num_workers = 4
    pin_memory = True
    drop_last = False
    data_loaders_generator = ECGRDVQLeaveOneOutDataLoadersGenerator(
        dir_paths=(train_dir, val_dir, test_dir),
        datasets_save_dir=datasets_save_dir,
        trajectory_length=trajectory_length,
        prediction_horizon=prediction_horizon,
        n_val_patients=n_val_patients,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    # Define the model
    m_dynamics = 5
    states_dim = 13
    units_factor = 2.0
    activation = 'relu'
    final_activation = None
    dropout = None
    bias = False
    n_layers_aux = 2
    model_params = {
        't': prediction_horizon,
        'input_dim': states_dim,
        'n_layers_encoder': n_encoder_layers,
        'n_layers_decoder': n_encoder_layers,
        'n_layers_aux': n_layers_aux,
        'l0_units': l0_units,
        'l0_units_aux': l0_units,
        'units_factor': units_factor,
        'units_factor_aux': units_factor,
        'activation': activation,
        'final_activation': final_activation,
        'dropout': dropout,
        'bias': bias,
        'k_prediction_steps': prediction_horizon,
    }

    # Define the optimizer
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

    # Define the loss & evaluation functions
    loss_hparams = {
        'alpha_1': 0.1,
        'alpha_2': 10e-7,
        's_p': prediction_horizon,
    }
    loss_fn = UniversalLinearEmbeddingLoss(
        **loss_hparams
    )

    evaluation_metric = MASELoss(
        m=prediction_horizon,
        trajectory_length=trajectory_length,
    )

    # Training specs
    num_epochs = 200
    checkpoints = True
    early_stopping = None
    checkpoints_mode = 'min'

    n_sets = len(data_loaders_generator)
    for i in range(16, n_sets):
        fold_dir = os.path.join(logs_dir, f'Fold_{i}')
        last_model_path = f"{fold_dir}/LastModel.PyTorchModule"

        if os.path.isdir(fold_dir) and os.path.isfile(last_model_path):
            continue

        # Get the data loaders
        train_dl, val_dl, test_dl = data_loaders_generator[i]

        # Define the model and optimizers
        model = UniversalLinearEmbeddingAE(
            **model_params
        )
        model.to(device)
        optimizers = [
            torch.optim.AdamW(
                params=model.parameters(),
                **optimizer_hparams,
            ),
        ]
        schedulers = [
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizers[-1],
                **scheduler_hparams
            ),
        ]
        optimizer = Optimizer(optimizers=optimizers, schedulers=schedulers)

        # Define the logger
        logger = Logger(
            log_dir=logs_dir,
            experiment_name=f'Fold_{i}',
            max_elements=1,
        )

        # Write Scenario Specs
        specs = {
            'Data Specs': '',
            "seed": seed,
            'trajectory_length': trajectory_length,
            'prediction_horizon': prediction_horizon,
            'DataLoader Specs': '',
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'drop_last': drop_last,
            'Model Specs': '',
            'Model': type(model).__name__,
        }
        specs.update(model_params)
        loss_params = {
            'Loss Specs': '',
            'loss_fn': f"{loss_fn}",
            'eval_fn': f"{evaluation_metric}",
            'Optimizer Specs': '',
            'optimizer': type(optimizers[0]).__name__,
        }
        specs.update(loss_params)
        specs.update(optimizer_hparams)
        specs['LR Scheduler Specs'] = ''
        specs['lr_scheduler'] = type(schedulers[0]).__name__
        specs.update(scheduler_hparams)

        specs_file = os.path.join(logs_dir, 'data_specs.txt')
        with open(specs_file, 'w') as f:
            for k, v in specs.items():
                f.write(f"{k}: {str(v)}\n")

        # Define the trainer
        trainer = LQTSTrainer(
            model=model,
            loss_fn=loss_fn,
            evaluation_metric=evaluation_metric,
            optimizer=optimizer,
            device=device,
            logger=logger,
        )

        print("Fitting the model")
        trainer.fit(
            dl_train=train_dl,
            dl_val=val_dl,
            num_epochs=num_epochs,
            checkpoints=checkpoints,
            checkpoints_mode=checkpoints_mode,
            early_stopping=early_stopping,
        )

        model = UniversalLinearEmbeddingAE(
            **model_params
        )
        model_logs_dir = os.path.join(logs_dir, f'Fold_{i}')
        model_ckpt_path = f"{model_logs_dir}/BestModel.PyTorchModule"  # loading best model
        model_ckp = torch.load(model_ckpt_path)
        model.load_state_dict(model_ckp['model'])
        model.to(device)
        trainer = LQTSTrainer(
            model=model,
            loss_fn=loss_fn,
            evaluation_metric=evaluation_metric,
            optimizer=optimizer,
            device=device,
            logger=logger,
        )

        # Evaluate
        trainer.evaluate(
            dl_test=test_dl,
            ignore_cap=True,
        )
