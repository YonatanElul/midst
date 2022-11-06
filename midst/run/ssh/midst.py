import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from datetime import datetime
from midst import DATA_DIR, LOGS_DIR
from midst.models.midst_model import MIDST
from midst.utils.loggers import Logger
from midst.optim.optim import Optimizer
from midst.data.datasets import SSTDataset
from midst.losses.losses import ModuleLoss, MASELoss
from midst.utils.trainers import InterrelatedDynamicsTrainer

import torch
import numpy as np

if __name__ == '__main__':
    # Set seed
    seed = 8783
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Define the log dir
    observable_dim = 64
    n_encoder_layers = 4
    l0_units = 64
    trajectory_length = 16
    prediction_horizon = 1
    residual_dynamics = True
    dmf_dynamics = True
    temporal_sharing = True
    separate_dynamics = False
    single_dynamics = False
    activation = 'leakyrelu'
    date = str(datetime.today()).split()[0]
    description = f"{'Residual_' if residual_dynamics else ''}{observable_dim}K_" \
                  f"{'DMF_' if dmf_dynamics else ''}" \
                  f"{'Sep_' if separate_dynamics else ''}" \
                  f"{'SingleDynamics_' if single_dynamics else ''}" \
                  f"{n_encoder_layers}E_" \
                  f"{trajectory_length}T_" \
                  f"{prediction_horizon}H_" \
                  f"{activation}"

    experiment_name = f"SST_MIDST_{description}_{date}"
    logs_dir = os.path.join(LOGS_DIR, experiment_name)
    os.makedirs(logs_dir, exist_ok=True)

    # Define the Datasets & Data loaders
    top_lat = (40, 90)
    bottom_lat = (60, 110)
    left_long = (190, 70)
    right_long = (210, 90)
    datasets_paths = (
        os.path.join(DATA_DIR, "SSTV2", "sst.wkmean.1990-present.nc"),
    )
    mask_path = os.path.join(DATA_DIR, "SSTV2", "lsmask.nc")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 8
    num_workers = 4
    val_ratio = 0.1
    test_ratio = 0.2
    train_ds = SSTDataset(
        mode='Train',
        datasets_paths=datasets_paths,
        surface_mask_path=mask_path,
        temporal_horizon=trajectory_length,
        prediction_horizon=prediction_horizon,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        top_lat=top_lat,
        bottom_lat=bottom_lat,
        left_long=left_long,
        right_long=right_long,
        single_dynamics=single_dynamics,
    )
    val_ds = SSTDataset(
        mode='Val',
        datasets_paths=datasets_paths,
        surface_mask_path=mask_path,
        temporal_horizon=trajectory_length,
        prediction_horizon=prediction_horizon,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        top_lat=top_lat,
        bottom_lat=bottom_lat,
        left_long=left_long,
        right_long=right_long,
        single_dynamics=single_dynamics,
    )

    pin_memory = True
    drop_last = False
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

    # Define the model
    if single_dynamics:
        m_dynamics = 1
        states_dim = train_ds.blocks.shape[1] * train_ds.blocks.shape[2]  # m systems * n states per system

    else:
        m_dynamics = train_ds.blocks.shape[1]
        states_dim = train_ds.blocks.shape[2]

    units_factor = 2
    dropout = 0
    final_activation = None
    bias = True
    n_forwards = 1
    identical_sigmas = False
    symmetric = False
    spectral_norm = False
    model_params = {
        'm_dynamics': m_dynamics,
        'observable_dim': observable_dim,
        'states_dim': states_dim,
        'n_encoder_layers': n_encoder_layers,
        'l0_units': l0_units,
        'units_factor': units_factor,
        'activation': activation,
        'final_activation': final_activation,
        'dropout': dropout,
        'bias': bias,
        'k_forward_prediction': prediction_horizon,
        'identical_sigmas': identical_sigmas,
        'symmetric': symmetric,
        'residual_dynamics': residual_dynamics,
        'dmf_dynamics': dmf_dynamics,
        'separate_dynamics': separate_dynamics,
        'spectral_norm': spectral_norm,
        'n_forwards': n_forwards,
        'temporal_sharing': temporal_sharing,
    }

    model = MIDST(
        **model_params
    )
    model.to(device)

    # Define the optimizer
    optimizer_hparams = {
        'lr': 0.001,
        'weight_decay': 1e-6,
    }
    optimizers = [
        torch.optim.AdamW(
            params=model.parameters(),
            **optimizer_hparams,
        ),
    ]
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
    schedulers = [
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizers[-1],
            **scheduler_hparams
        ),
    ]
    optimizer = Optimizer(optimizers=optimizers, schedulers=schedulers)

    # Define the loss & evaluation functions
    loss_fn = ModuleLoss(
        model=torch.nn.MSELoss(),
        scale=1,
    )

    evaluation_metric = MASELoss(
        m=prediction_horizon,
        trajectory_length=trajectory_length,
    )

    # Define the logger
    logger = Logger(
        log_dir=LOGS_DIR,
        experiment_name=experiment_name,
        max_elements=2,
    )

    # Define the trainer
    num_epochs = 200
    checkpoints = True
    early_stopping = None
    checkpoints_mode = 'min'
    trainer = InterrelatedDynamicsTrainer(
        model=model,
        loss_fn=loss_fn,
        evaluation_metric=evaluation_metric,
        optimizer=optimizer,
        device=device,
        logger=logger,
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
        'Trainer Specs': '',
        'num_epochs': num_epochs,
        'checkpoints': checkpoints,
        'early_stopping': early_stopping,
        'checkpoints_mode': checkpoints_mode,
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

    print("Fitting the model")
    trainer.fit(
        dl_train=train_dl,
        dl_val=val_dl,
        num_epochs=num_epochs,
        checkpoints=checkpoints,
        checkpoints_mode=checkpoints_mode,
        early_stopping=early_stopping,
    )

    # Define the test-set
    print("Evaluating over the test set")
    test_ds = SSTDataset(
        mode='Test',
        datasets_paths=datasets_paths,
        surface_mask_path=mask_path,
        temporal_horizon=trajectory_length,
        prediction_horizon=prediction_horizon,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        top_lat=top_lat,
        bottom_lat=bottom_lat,
        left_long=left_long,
        right_long=right_long,
        single_dynamics=single_dynamics,
    )
    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    model = MIDST(
        **model_params
    )
    model_ckpt_path = f"{logs_dir}/BestModel.PyTorchModule"  # loading best model
    model_ckp = torch.load(model_ckpt_path)
    model.load_state_dict(model_ckp['model'])
    model.to(device)
    trainer = InterrelatedDynamicsTrainer(
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
