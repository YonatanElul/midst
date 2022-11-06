from abc import ABC
from torch import nn
from midst.utils.defaults import (
    MODELS_TENSOR_PREDICITONS_KEY,
    GT_TENSOR_PREDICITONS_KEY,
    GT_TENSOR_INPUTS_KEY, OTHER_KEY,
)
from midst.utils.loggers import Logger
from midst.optim.optim import Optimizer
from midst.losses.losses import LossComponent
from torch.utils.data.dataloader import DataLoader
from typing import List, Callable, Tuple, Sequence, Optional

import os
import tqdm
import torch
import numpy as np


class BaseTrainer(ABC):
    """
    A class abstracting the various tasks of training models.
    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """

    def __init__(
            self,
            model: nn.Module,
            loss_fn: LossComponent,
            evaluation_metric: LossComponent,
            optimizer: Optimizer,
            logger: Logger,
            device: torch.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            ),
            max_iterations_per_epoch: int = float('inf'),
            loss_params_generator_per_epoch: Optional[Callable] = None,
    ):
        """
        Initialize the trainer.

        :param model: Instance of the _model to train.
        :param loss_fn: A LossComponent object, which serves as the loss
        function to evaluate with.
        :param evaluation_metric: A LossComponent object,
        which takes in the predictions and ground truth, and returns float,
        representing an 'accuracy' score.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        :param max_iterations_per_epoch: Upper limit on the number of iterations (i.e.
        batches) to compute in each epoch.
        :param logger: (Logger) A logger for saving all the required
        intermediate results.
        :param loss_params_generator_per_epoch: (Callable) A callable method which takes as input the epoch number
        and outputs parameters updated parameters for the loss function (such as regularization weights etc.)
        """

        self._model = model.to(device)
        self._loss_fn = loss_fn
        self._evaluation_metric = evaluation_metric
        self._optimizer = optimizer
        self._device = device
        self._max_iterations_per_epoch = max_iterations_per_epoch
        self._logger = logger
        self._save_path_dir = logger.save_dir
        self._loss_params_generator_per_epoch = loss_params_generator_per_epoch

    def fit(
            self,
            dl_train: DataLoader,
            dl_val: DataLoader,
            num_epochs: int = 10,
            checkpoints: bool = False,
            checkpoints_mode: str = 'min',
            early_stopping: int = None,
    ) -> Tuple[Sequence[float], Sequence[float],
               Sequence[float], Sequence[float]]:
        """
        Trains the _model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.

        :param dl_train: Dataloader for the training set.
        :param dl_val: Dataloader for the validation set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save _model to file every time the
            tests set accuracy improves. Should be a string containing a
            filename without extension.
        :param checkpoints_mode: (str) Whether to optimize for minimum, or maximum,
        score.
        :param early_stopping: Whether to stop training early if there is no
            tests loss improvement for this number of epochs.

        :return: A tuple with four lists, containing, in that order, the train loss and
         accuracy and the tests loss and accuracy.
        """

        # Saving an initial checkpoint
        self._logger.clear_logs()
        saved_object = {
            'model': self._model.state_dict(),
        }
        saving_path = os.path.join(
            self._save_path_dir, f"BestModel.PyTorchModule"
        )
        torch.save(
            obj=saved_object,
            f=saving_path
        )

        best_acc = None
        epochs_without_improvement = 0
        train_loss, train_acc, eval_loss, eval_acc = [], [], [], [],
        for epoch in range(num_epochs):
            print(f'\n--- EPOCH {epoch + 1}/{num_epochs} ---')

            if self._loss_params_generator_per_epoch is not None:
                loss_params = self._loss_params_generator_per_epoch(epoch)
                self._loss_fn.update(params=loss_params)

            loss, acc = self.train_epoch(dl_train=dl_train)
            train_loss.append(loss)
            train_acc.append(acc)
            self._logger.log_variable(loss, f"train_loss")
            self._logger.log_variable(acc, f"train_acc")

            # Run an evaluation of the _model & save the results
            loss, acc = self.test_epoch(dl_test=dl_val, ignore_cap=False)
            eval_loss.append(loss)
            eval_acc.append(acc)
            self._logger.log_variable(loss, f"eval_loss")
            self._logger.log_variable(acc, f"eval_acc")

            # Perform the schedulers step - if relevant
            self._optimizer.schedulers_step(np.mean(acc).item())

            if epoch == 0:
                best_acc = acc
                epochs_without_improvement = 0
                save_checkpoint = True

            else:
                if checkpoints_mode == 'max' and acc > best_acc:
                    best_acc = acc
                    save_checkpoint = True
                    epochs_without_improvement = 0

                elif checkpoints_mode == 'min' and acc < best_acc:
                    best_acc = acc
                    save_checkpoint = True
                    epochs_without_improvement = 0

                else:
                    save_checkpoint = False
                    epochs_without_improvement += 1

            # Create a checkpoint after each epoch if applicable
            if checkpoints and save_checkpoint:
                print(f"\nSaving Checkpont at epoch {epoch + 1}")

                saved_object = {
                    'model': self._model.state_dict(),
                }
                saving_path = os.path.join(
                    self._save_path_dir, f"BestModel.PyTorchModule"
                )
                torch.save(
                    obj=saved_object,
                    f=saving_path
                )
                self._logger.flush(
                    variables=[n for n in self._logger.logged_vars if 'train' in n],
                    save_dir=self._logger.save_dir_train,
                )
                self._logger.flush(
                    variables=[n for n in self._logger.logged_vars if 'eval' in n],
                    save_dir=self._logger.save_dir_val,
                )

            # We haven't improved at all in the last 'early_stopping' epochs
            if (
                    early_stopping is not None and
                    epochs_without_improvement == early_stopping
            ):
                print(f"\nSaving Checkpont at epoch {epoch + 1}")
                saved_object = {
                    'model': self._model.state_dict(),
                }
                saving_path = os.path.join(
                    self._save_path_dir, f"Checkpoint_Epoch_{epoch}.PyTorchModule"
                )
                torch.save(
                    obj=saved_object,
                    f=saving_path
                )
                self._logger.flush(
                    variables=[n for n in self._logger.logged_vars if 'train' in n],
                    save_dir=self._logger.save_dir_train,
                )
                self._logger.flush(
                    variables=[n for n in self._logger.logged_vars if 'eval' in n],
                    save_dir=self._logger.save_dir_val,
                )

                print("Reached the Early Stop condition.\nStopping the training.")
                break

        print(f"\nSaving the final Checkpont")
        saved_object = {
            'model': self._model.state_dict(),
        }
        saving_path = os.path.join(
            self._save_path_dir, "LastModel.PyTorchModule"
        )
        torch.save(
            obj=saved_object,
            f=saving_path
        )

        self._logger.log_variable(train_loss, "fit_train_loss", ignore_cap=True)
        self._logger.log_variable(train_acc, "fit_train_acc", ignore_cap=True)
        self._logger.log_variable(eval_loss, "fit_eval_loss", ignore_cap=True)
        self._logger.log_variable(eval_acc, "fit_eval_acc", ignore_cap=True)

        self._logger.flush(
            variables=[n for n in self._logger.logged_vars if 'train' in n],
            save_dir=self._logger.save_dir_train,
        )
        self._logger.flush(
            variables=[n for n in self._logger.logged_vars if 'eval' in n],
            save_dir=self._logger.save_dir_val,
        )

        return train_loss, train_acc, eval_loss, eval_acc

    def evaluate(
            self,
            dl_test: DataLoader,
            ignore_cap: bool = False,
    ) -> Sequence[float]:
        """
        Run a single evaluation epoch on an held out test set.

        :param dl_test: Dataloader for the test set.
        :param ignore_cap: (bool) Whether to ignore the logger cap or not.
        """

        print(f'\n--- Evaluating Test Set ---')
        self._logger.clear_logs()
        loss, acc = self.test_epoch(dl_test=dl_test, ignore_cap=ignore_cap)
        self._logger.log_variable(loss, f"eval_loss", ignore_cap=ignore_cap)
        self._logger.log_variable(acc, f"eval_acc", ignore_cap=ignore_cap)
        self._logger.flush(
            variables=[n for n in self._logger.logged_vars if 'eval' in n],
            save_dir=self._logger.save_dir_test,
        )

        return loss, acc

    def forecast(
            self,
            ds_test: torch.utils.data.Dataset,
            horizon: int = 1,
            starting_ind: int = 0,
            noise: float = 0.0
    ) -> None:
        print(f'\n--- Forecasting Future Values ---')
        self._logger.clear_logs()
        self.forecast_epoch(ds_test=ds_test, horizon=horizon, starting_ind=starting_ind, noise=noise)
        self._logger.flush(
            variables=[n for n in self._logger.logged_vars if 'forecast' in n],
            save_dir=self._logger.save_dir_test,
        )

    def train_epoch(self, dl_train: DataLoader) -> Tuple[float, float]:
        """
        Train once over a training set (single epoch).

        :param dl_train: DataLoader for the training set.

        :return: A tuple containing the aggregated training epoch loss and accuracy
        """

        self._model.train(True)
        loss, accuracy = self._foreach_batch(
            dl_train,
            self.train_batch,
            max_iterations_per_epoch=self._max_iterations_per_epoch,
        )

        return np.mean(loss).item(), np.mean(accuracy).item()

    def test_epoch(self, dl_test: DataLoader, ignore_cap: bool = False) -> Tuple[float, float]:
        """
        Evaluate a model once over a tests set (single epoch).

        :param dl_test: DataLoader for the tests set.
        :param ignore_cap: (bool) Whether to ignore the logger cap or not.

        :return: A tuple containing the aggregated tests epoch loss and accuracy
        """

        self._model.train(False)
        loss, accuracy = self._foreach_batch(
            dl=dl_test,
            forward_fn=self.test_batch,
            ignore_cap=ignore_cap,
            max_iterations_per_epoch=self._max_iterations_per_epoch,
        )

        return np.mean(loss).item(), np.mean(accuracy).item()

    def forecast_epoch(
            self,
            ds_test: torch.utils.data.Dataset,
            horizon: int = 1,
            starting_ind: int = 0,
            noise: float = 0.0,
    ) -> None:
        self._model.train(False)
        batch_0 = ds_test[starting_ind]

        x = batch_0[GT_TENSOR_INPUTS_KEY][None, ...]
        y = batch_0[GT_TENSOR_PREDICITONS_KEY][None, ...]
        trajectory_length = x.shape[2]
        forecasting_horizon = y.shape[2] // trajectory_length
        for h in range(horizon):
            print(f"Predicting step {(h + 1)} / {horizon}")
            if noise:
                xs = [
                    self.forecast_batch(
                        x=(x + torch.normal(0, noise, x.shape, device=x.device)),
                        y=y,
                    )[None, ...]
                    for _ in range(10)
                ]
                x = torch.cat(xs, 0)
                x = x.mean(0)

            else:
                x = self.forecast_batch(
                    x,
                    y=y,
                )

            # In case the prediction horizon is greater than 1
            x = x[..., -trajectory_length:, :]

            ind = starting_ind + ((h + 1) * forecasting_horizon)
            if ind < len(ds_test):
                batch = ds_test[ind]
                y = batch[GT_TENSOR_PREDICITONS_KEY][None, ...]

            else:
                y = None
                continue

    def train_batch(self, batch, ignore_cap: bool = False) -> Tuple[float, float]:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.

        :param batch: A dict, representing a single batch of data with the keys,
        'x' and 'y' representing the inputs and ground truth outputs.
        :param ignore_cap: (bool) Whether to ignore the logger cap or not.

        :return: A tuple containing the train loss and accuracy on the current batch
        """

        # Unpack the batch
        x = batch[MODELS_TENSOR_PREDICITONS_KEY]
        x = x.to(self._device)

        # Run the forward pass
        outputs = self._model.forward(x)

        # Zero the gradients after each step
        self._optimizer.zero_grad()

        # Compute the loss with respect to the true labels
        batch[MODELS_TENSOR_PREDICITONS_KEY] = outputs
        loss = self._loss_fn(batch)

        # Run the backwards pass
        loss.backward()

        # Perform the optimization step
        self._optimizer.step()

        # Compute the 'accuracy'
        accuracy = self._evaluation_metric(batch).item()

        return loss.item(), accuracy

    def test_batch(self, batch, ignore_cap: bool = False) -> Tuple[float, float]:
        """
        Runs a single batch forward through the model, and calculates loss and accuracy.

        :param batch: A dict, representing a single batch of data with the keys,
        'x' and 'y' representing the inputs and ground truth outputs.
        :param ignore_cap: (bool) Whether to ignore the logger cap or not.

        :return: A tuple containing the tests loss and accuracy on the current batch
        """

        # Unpack the batch
        x = batch[MODELS_TENSOR_PREDICITONS_KEY]
        x = x.to(self._device)

        with torch.no_grad():
            # Run the forward pass
            outputs = self._model.forward(x)

            # Compute the loss with respect to the true labels
            batch[MODELS_TENSOR_PREDICITONS_KEY] = outputs
            loss = self._loss_fn(batch)

            # Compute the 'accuracy'
            accuracy = self._evaluation_metric(batch).item()

        return loss.item(), accuracy

    def forecast_batch(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:

        x = x.to(self._device)

        with torch.no_grad():
            # Run the forward pass
            outputs = self._model.forward(x)
            next_x = outputs[MODELS_TENSOR_PREDICITONS_KEY]

            self._logger.log_variable(
                x.detach().cpu().numpy(),
                f"x_gt_forecast",
                ignore_cap=True,
            )
            self._logger.log_variable(
                next_x.detach().cpu().numpy(),
                f"y_pred_forecast",
                ignore_cap=True,
            )

            if y is not None:
                self._logger.log_variable(
                    y.detach().cpu().numpy(),
                    f"y_gt_forecast",
                    ignore_cap=True,
                )

        return next_x

    @staticmethod
    def _foreach_batch(
            dl: DataLoader,
            forward_fn: Callable,
            ignore_cap: bool = False,
            max_iterations_per_epoch: Optional[int] = None,
    ) -> Tuple[List, List]:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.

        :param dl: The DataLoader object from which to query batches.
        :param forward_fn: The forward method to apply to each batch,
        i.e. `train_batch` or `test_batch`.
        :param ignore_cap: (bool) Whether to ignore the logger cap or not.

        :return: A tuple of two lists, the first contains the losses over all batches in
        the current epoch, and the second ones contains all of the accuracies.
        """

        losses = []
        accuracies = []
        num_batches = len(dl.batch_sampler)

        if max_iterations_per_epoch is not None:
            num_batches = min(num_batches, max_iterations_per_epoch)

        pbar_name = forward_fn.__name__
        dl_iter = iter(dl)
        with tqdm.tqdm(desc=pbar_name, total=num_batches) as pbar:
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                loss, acc = forward_fn(data, ignore_cap)

                pbar.set_description(f'{pbar_name} ({loss:.3f})')
                pbar.update()

                losses.append(loss)
                accuracies.append(acc)

            avg_loss = np.mean(losses).item()
            avg_acc = np.mean(accuracies).item()
            pbar.set_description(
                f'{pbar_name} (Avg. Loss {avg_loss:.3f}, Avg. Accuracy {avg_acc:.3f})'
            )

        return losses, accuracies


class ConcurrentDynamicsTrainer(BaseTrainer):
    """
    A trainer for concurrent-dynamics related experiments and models.
    """

    def __init__(
            self,
            model: nn.Module,
            loss_fn: LossComponent,
            evaluation_metric: LossComponent,
            optimizer: Optimizer,
            logger: Logger,
            device: torch.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            ),
            max_iterations_per_epoch: int = float('inf'),
            loss_params_generator_per_epoch: Optional[Callable] = None,
    ):
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            evaluation_metric=evaluation_metric,
            optimizer=optimizer,
            device=device,
            max_iterations_per_epoch=max_iterations_per_epoch,
            logger=logger,
            loss_params_generator_per_epoch=loss_params_generator_per_epoch,
        )

    def train_batch(self, batch, ignore_cap: bool = False) -> Tuple[float, float]:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.

        :param batch: A dict, representing a single batch of data with the keys,
        'x' and 'y' representing the sequences of inputs and ground truth outputs.
        :param ignore_cap: (bool) Whether to ignore the logger cap or not.

        :return: A tuple containing the train loss and accuracy on the current batch
        """

        # Unpack the batch
        x = batch[GT_TENSOR_INPUTS_KEY].to(self._device)
        y = batch[GT_TENSOR_PREDICITONS_KEY].to(self._device)

        # Run the forward pass
        outputs = self._model.forward(x)

        # Log trajectories
        self._logger.log_variable(
            y.detach().cpu().numpy(),
            "y_gt_train",
            ignore_cap=ignore_cap,
        )
        self._logger.log_variable(
            outputs[MODELS_TENSOR_PREDICITONS_KEY].detach().cpu().numpy(),
            "y_pred_train",
            ignore_cap=ignore_cap,
        )

        # Zero the gradients after each step
        self._optimizer.zero_grad()

        # Compute the loss with respect to the true labels
        outputs[GT_TENSOR_PREDICITONS_KEY] = y
        outputs[GT_TENSOR_INPUTS_KEY] = x

        loss = self._loss_fn(outputs)

        # Run the backwards pass
        loss.backward()

        # Perform the optimization step
        self._optimizer.step()

        # Compute the 'accuracy'
        accuracy = self._evaluation_metric(outputs)

        return loss.item(), accuracy.item()

    def test_batch(self, batch, ignore_cap: bool = False) -> Tuple[float, float]:
        """
        Runs a single batch forward through the model, and calculates loss and accuracy.

        :param batch: A dict, representing a single batch of data with the keys,
        'x' and 'y' representing the sequences of inputs and ground truth outputs.
        :param ignore_cap: (bool) Whether to ignore the logger cap or not.

        :return: A tuple containing the tests loss and accuracy on the current batch
        """

        # Unpack the batch
        x = batch[GT_TENSOR_INPUTS_KEY].to(self._device)
        y = batch[GT_TENSOR_PREDICITONS_KEY].to(self._device)

        with torch.no_grad():
            # Run the forward pass
            outputs = self._model.forward(x)

            # Log trajectories
            self._logger.log_variable(
                y.detach().cpu().numpy(),
                "y_gt_eval",
                ignore_cap=ignore_cap,
            )
            self._logger.log_variable(
                x.detach().cpu().numpy(),
                "x_gt_eval",
                ignore_cap=ignore_cap,
            )
            self._logger.log_variable(
                outputs[MODELS_TENSOR_PREDICITONS_KEY].detach().cpu().numpy(),
                "y_pred_eval",
                ignore_cap=ignore_cap,
            )

            if OTHER_KEY in outputs and 'dynamics' in outputs[OTHER_KEY] and ignore_cap:
                if isinstance(outputs[OTHER_KEY]['dynamics'][0], list):
                    [
                        [
                            self._logger.log_variable(
                                outputs[OTHER_KEY]['dynamics'][t][m].detach().cpu().numpy(),
                                f"eval_dynamics_{m}_t_{t}",
                                ignore_cap=True,
                            )
                            for m in range(len(outputs[OTHER_KEY]['dynamics'][0]))
                        ]
                        for t in range(len(outputs[OTHER_KEY]['dynamics']))
                    ]

                else:
                    [
                        self._logger.log_variable(
                            outputs[OTHER_KEY]['dynamics'][t].detach().cpu().numpy(),
                            f"eval_dynamics_{t}",
                            ignore_cap=True,
                        )
                        for t in range(len(outputs[OTHER_KEY]['dynamics']))
                    ]

            if OTHER_KEY in outputs and 'U_per_t' in outputs[OTHER_KEY] and ignore_cap:
                if isinstance(outputs[OTHER_KEY]['U_per_t'][0], torch.Tensor):
                    [
                        self._logger.log_variable(
                            outputs[OTHER_KEY]['U_per_t'][t].detach().cpu().numpy(),
                            f"eval_U_per_t_{t}",
                            ignore_cap=True,
                        )
                        for t in range(len(outputs[OTHER_KEY]['U_per_t']))
                    ]
                    [
                        self._logger.log_variable(
                            outputs[OTHER_KEY]['V_per_t'][t].detach().cpu().numpy(),
                            f"eval_V_per_t_{t}",
                            ignore_cap=True,
                        )
                        for t in range(len(outputs[OTHER_KEY]['V_per_t']))
                    ]

                else:
                    [
                        [
                            self._logger.log_variable(
                                outputs[OTHER_KEY]['U_per_t'][t][m].detach().cpu().numpy(),
                                f"eval_U_per_t_{t}_m_{m}",
                                ignore_cap=True,
                            )
                            for m in range(len(outputs[OTHER_KEY]['U_per_t'][0]))
                        ]
                        for t in range(len(outputs[OTHER_KEY]['U_per_t']))
                    ]
                    [
                        [
                            self._logger.log_variable(
                                outputs[OTHER_KEY]['V_per_t'][t][m].detach().cpu().numpy(),
                                f"eval_V_per_t_{t}_m_{m}",
                                ignore_cap=True,
                            )
                            for m in range(len(outputs[OTHER_KEY]['V_per_t'][0]))
                        ]
                        for t in range(len(outputs[OTHER_KEY]['V_per_t']))
                    ]

                [
                    [
                        self._logger.log_variable(
                            outputs[OTHER_KEY]['S_per_t'][t][m].detach().cpu().numpy(),
                            f"eval_S_per_t_{t}_per_m_{m}",
                            ignore_cap=True,
                        )
                        for m in range(len(outputs[OTHER_KEY]['S_per_t'][0]))
                    ]
                    for t in range(len(outputs[OTHER_KEY]['S_per_t']))
                ]

            # Compute the loss with respect to the true labels
            outputs[GT_TENSOR_PREDICITONS_KEY] = y
            outputs[GT_TENSOR_INPUTS_KEY] = x

            loss = self._loss_fn(outputs)

            # Compute the 'accuracy'
            accuracy = self._evaluation_metric(outputs)

        return loss.item(), accuracy.item()


class ECGRDVQTrainer(ConcurrentDynamicsTrainer):
    """
    A trainer used specifically for the medication effects experiments done on the
     ECGEDVQ dataset
    """

    def __init__(
            self,
            model: nn.Module,
            loss_fn: LossComponent,
            evaluation_metric: LossComponent,
            optimizer: Optimizer,
            logger: Logger,
            device: torch.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            ),
            max_iterations_per_epoch: int = float('inf'),
    ):
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            evaluation_metric=evaluation_metric,
            optimizer=optimizer,
            device=device,
            max_iterations_per_epoch=max_iterations_per_epoch,
            logger=logger,
        )

        self._ignore_cap = False

    def train_batch(self, batch, ignore_cap: bool = False) -> Tuple[float, float]:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.

        :param batch: A dict, representing a single batch of data with the keys,
        'x' and 'y' representing the sequences of inputs and ground truth outputs.
        :param ignore_cap: (bool) Whether to ignore the logger cap or not.

        :return: A tuple containing the train loss and accuracy on the current batch
        """

        # Unpack the batch
        x = batch[GT_TENSOR_INPUTS_KEY].to(self._device)
        y = batch[GT_TENSOR_PREDICITONS_KEY].to(self._device)

        # Run the forward pass
        outputs = self._model.forward(x)

        # Log trajectories
        self._logger.log_variable(
            y.detach().cpu().numpy(),
            "y_gt_train",
            ignore_cap=ignore_cap,
        )
        self._logger.log_variable(
            outputs[MODELS_TENSOR_PREDICITONS_KEY].detach().cpu().numpy(),
            "y_pred_train",
            ignore_cap=ignore_cap,
        )

        # Zero the gradients after each step
        self._optimizer.zero_grad()

        # Compute the loss with respect to the true labels
        outputs[GT_TENSOR_PREDICITONS_KEY] = y
        outputs[GT_TENSOR_INPUTS_KEY] = x

        loss = self._loss_fn(outputs)

        # Run the backwards pass
        loss.backward()

        # Perform the optimization step
        self._optimizer.step()

        # Compute the 'accuracy'
        accuracy = self._evaluation_metric(outputs)

        return loss.cpu().detach().item(), accuracy.cpu().detach().item()

    def test_batch(self, batch: dict, ignore_cap: bool = False) -> Tuple[float, float]:
        """
        Runs a single batch forward through the model, and calculates loss and accuracy.

        :param batch: A dict, representing a single batch of data with the keys,
        'x' and 'y' representing the sequences of inputs and ground truth outputs.
        :param ignore_cap: (bool) Whether to ignore the logger cap or not.

        :return: A tuple containing the tests loss and accuracy on the current batch
        """

        # Unpack the batch
        x = batch[GT_TENSOR_INPUTS_KEY].to(self._device)
        y = batch[GT_TENSOR_PREDICITONS_KEY].to(self._device)

        with torch.no_grad():
            # Run the forward pass
            outputs = self._model(x)

            # Log trajectories
            self._logger.log_variable(
                y.detach().cpu().numpy(),
                "y_gt_eval",
                ignore_cap=ignore_cap,
            )
            self._logger.log_variable(
                x.detach().cpu().numpy(),
                "x_gt_eval",
                ignore_cap=ignore_cap,
            )
            self._logger.log_variable(
                outputs[MODELS_TENSOR_PREDICITONS_KEY].detach().cpu().numpy(),
                "y_pred_eval",
                ignore_cap=ignore_cap,
            )

            if OTHER_KEY in outputs and 'dynamics' in outputs[OTHER_KEY] and ignore_cap:
                if isinstance(outputs[OTHER_KEY]['dynamics'][0], list):
                    [
                        [
                            self._logger.log_variable(
                                outputs[OTHER_KEY]['dynamics'][t][m].detach().cpu().numpy(),
                                f"eval_dynamics_{m}_t_{t}",
                                ignore_cap=ignore_cap,
                            )
                            for m in range(len(outputs[OTHER_KEY]['dynamics'][0]))
                        ]
                        for t in range(len(outputs[OTHER_KEY]['dynamics']))
                    ]

                else:
                    [
                        self._logger.log_variable(
                            outputs[OTHER_KEY]['dynamics'][t].detach().cpu().numpy(),
                            f"eval_dynamics_{t}",
                            ignore_cap=True,
                        )
                        for t in range(len(outputs[OTHER_KEY]['dynamics']))
                    ]

            # Compute the loss & accuracy
            outputs[GT_TENSOR_PREDICITONS_KEY] = y
            outputs[GT_TENSOR_INPUTS_KEY] = x

            loss = self._loss_fn(outputs).cpu().detach().item()
            accuracy = self._evaluation_metric(outputs).cpu().detach().item()

        return loss, accuracy
