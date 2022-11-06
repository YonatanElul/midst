from numpy import array
from torch import float32
from abc import ABC, abstractmethod
from torch.nn.functional import mse_loss
from typing import Sequence, Optional, Union, Dict
from torch import nn, Tensor, cat, from_numpy, eye, matmul
from midst.utils.defaults import (
    MODELS_TENSOR_PREDICITONS_KEY,
    GT_TENSOR_PREDICITONS_KEY,
    MODELS_SEQUENCE_PREDICITONS_KEY,
    GT_SEQUENCE_PREDICITONS_KEY,
    GT_TENSOR_INPUTS_KEY,
    OTHER_KEY,
)

import torch


class LossComponent(nn.Module, ABC):
    """
    An abstract API class for loss models
    """

    def __init__(self):
        super(LossComponent, self).__init__()

    @abstractmethod
    def forward(self, inputs: Union[Dict, Sequence]) -> Tensor:
        """
        The forward logic of the loss class.

        :param inputs: (Dict) Either a dictionary with the predictions from the forward pass and the ground truth
        outputs. The possible keys are specified by the following variables:
            MODELS_TENSOR_PREDICITONS_KEY
            MODELS_SEQUENCE_PREDICITONS_KEY
            GT_TENSOR_PREDICITONS_KEY
            GT_SEQUENCE_PREDICITONS_KEY
            GT_TENSOR_INPUTS_KEY
            GT_SEQUENCE_INPUTS_KEY
            OTHER_KEY

        Which can be found under .../DynamicalSystems/utils/defaults.py

        Or a Sequence of dicts, each where each element is a dict with the above-mentioned structure.

        :return: (Tensor) A scalar loss.
        """

        raise NotImplemented

    def __call__(self, inputs: Union[Dict, Sequence]) -> Tensor:
        return self.forward(inputs=inputs)

    def update(self, params: Dict[str, any]):
        for key in params:
            if hasattr(self, key):
                setattr(self, key, params[key])


class ModuleLoss(LossComponent):
    """
    A LossComponent which takes in a PyTorch loss Module and decompose the inputs according to the module's
    expected API.
    """

    def __init__(self, model: nn.Module, scale: float = 1.0):
        """
        The constructor for the ModuleLoss class
        :param model: (PyTorch Module) The loss model, containing the computation logic.
        :param model: (float) Scaling factor for the loss.
        """

        super().__init__()

        self.model = model
        self.scale = scale

    def forward(self, inputs: Dict) -> Tensor:
        """
        Basically a wrapper around the forward of the inner model, which decompose the inputs to the expected
        structure expected by the PyTorch module.

        :param inputs: (dict) The outputs of the forward pass of the model along with the ground-truth labels.
        :return: (Tensor) A scalar Tensor representing the aggregated loss
        """

        y_pred = inputs[MODELS_TENSOR_PREDICITONS_KEY]
        y = inputs[GT_TENSOR_PREDICITONS_KEY]

        loss = self.scale * self.model(y, y_pred)

        return loss


class SequenceLossComponent(LossComponent):
    """
    A base loss class for losses which operates on sequences
    """

    AGG_METHODS = (
        'sum',
        'avg',
    )

    def __init__(self, loss_model: Union[nn.Module, LossComponent],
                 aggregation_method: str = 'sum',
                 discount_rate: float = 1.):
        """
        Constructor for the SequenceLossComponen class.

        :param loss_model: (Union[nn.Module, LossComponent]) - The loss model to apply to each element in the sequence.
        :param aggregation_method: (str) Specify the method for aggregating the loss across the different elements in
        the sequence.
        :param discount_rate: (float) By how much to discount the loss of each element i in the sequence.
        The i-th element will have a weight of discount_rate^i. Defaults to 1.0, i.e. no discount.

        Viable aggregation methods are specified in the AGG_METHODS class constant.
        """

        super().__init__()

        assert aggregation_method in self.AGG_METHODS, f"{aggregation_method} is not a viable aggregation method." \
                                                       f"please use one of the following: {self.AGG_METHODS}."

        self._model = loss_model
        self._aggregation_method = aggregation_method
        self._discount_rate = discount_rate

    @staticmethod
    def _aggregate_sum(losses: Tensor) -> Tensor:
        """
        Utility method for summing all losses from all elements in the sequence.
        """

        return losses.sum()

    @staticmethod
    def _aggregate_avg(losses: Tensor) -> Tensor:
        """
        Utility method for averaging all losses from all elements in the sequence.
        """

        return losses.mean()

    def _aggregate(self, losses: Sequence[Tensor]) -> Tensor:
        """
        Utility method for aggregating all losses from all elements in the sequence.

        :param losses: (Sequence[Tensor]) The losses of each element to be aggregated.

        :return: (Tensor) The aggregated loss
        """

        assert isinstance(losses, list) or isinstance(losses, tuple), \
            f"The 'losses' param to '_aggregate' in  'SequenceLossComponent' " \
            f"must be a sequential type, i.e. tuple or list, not {type(losses)}"

        discounts = from_numpy(
            array([self._discount_rate ** i for i in range(1, len(losses) + 1)])).type(
            float32)
        losses = cat(losses, dim=0)
        losses = losses * discounts.to(losses.device)

        if self._aggregation_method == 'sum':
            return self._aggregate_sum(losses=losses)

        elif self._aggregation_method == 'avg':
            return self._aggregate_avg(losses=losses)

    def forward(self, inputs: Union[Dict, Sequence]) -> Tensor:
        x_seq = inputs[MODELS_SEQUENCE_PREDICITONS_KEY]
        y_seq = inputs[GT_SEQUENCE_PREDICITONS_KEY]

        assert len(x_seq) == len(y_seq), \
            f"the '{MODELS_SEQUENCE_PREDICITONS_KEY}' and '{GT_SEQUENCE_PREDICITONS_KEY}' keys in the 'inputs' param" \
            f"in the 'forward' call of SequenceLossComponent must be represent sequences of identical length. " \
            f"However, they returned a sequences of lengths {len(x_seq)}, {len(y_seq)} respectively."

        losses = [
            self._model(
                {
                    MODELS_TENSOR_PREDICITONS_KEY: x_seq[i],
                    GT_TENSOR_PREDICITONS_KEY: y_seq[i],
                }
            ).unsqueeze(0)
            for i in range(len(x_seq))
        ]
        loss = self._aggregate(losses=losses)

        return loss


class CompoundedLoss(LossComponent):
    """
    A wrapper class for handling multiple loss functions which should be applied
    together.
    """

    def __init__(
            self,
            losses: Sequence[LossComponent],
            losses_weights: Optional[Sequence[float]] = None):
        """
        Constructor for the CompoundedLoss class.

        :param losses: (Sequence[LossComponent]) A sequence of loss
        modules to be applied.
        :param losses_weights: (Optional) A sequence of loss weights
        to be applied to each loss module.
        """

        if losses_weights is not None:
            assert len(losses) == len(losses_weights), \
                f"losses_weights should either specify a weight of type float for " \
                f"each loss in losses, or be left as None. Currently there are" \
                f"{len(losses)} and {len(losses_weights)} weights."

            self._losses_weights = losses_weights

        else:
            self._losses_weights = [1.0 for _ in range(len(losses))]

        super(CompoundedLoss, self).__init__()

        self._losses = losses

    def forward(self, inputs: Union[Dict, Sequence]) -> Tensor:
        loss = sum([self._losses_weights[i] * self._losses[i](inputs)
                    for i in range(len(self._losses))])

        return loss


class SingleSystemModuleLoss(ModuleLoss):
    """
    A LossComponent which takes in a PyTorch loss Module and decompose the inputs according to the module's
    expected API, for a single system.
    """

    def __init__(self, system_ind: int, model: nn.Module):
        """
        The constructor for the ModuleLoss class
        :param model: (PyTorch Module) The loss model, containing the computation logic.
        """

        super().__init__(model)
        self._system_ind = system_ind

    def forward(self, inputs: Dict) -> Tensor:
        """
        Basically a wrapper around the forward of the inner model, which decompose the inputs to the expected
        structure expected by the PyTorch module.

        :param inputs: (dict) The outputs of the forward pass of the model along with the ground-truth labels.
        :return: (Tensor) A scalar Tensor representing the aggregated loss
        """

        y_pred = inputs[MODELS_TENSOR_PREDICITONS_KEY]
        y = inputs[GT_TENSOR_PREDICITONS_KEY][:, self._system_ind]

        loss = self._model(y, y_pred)

        return loss


class UniversalLinearEmbeddingLoss(LossComponent):
    """
    The combined loss criterion used in https://www.nature.com/articles/s41467-018-07210-0.pdf.
    """

    def __init__(self, alpha_1: float, alpha_2: float, s_p: int):
        """
        :param alpha_1: (float) Weight of the recon & pred losses
        :param alpha_2: (float) Weight of the inf norm loss
        :param s_p: (int) A hyper-parameter for how many steps to check in the prediction loss.
        """

        super(UniversalLinearEmbeddingLoss, self).__init__()

        self._alpha_1 = alpha_1
        self._alpha_2 = alpha_2
        self._s_p = s_p

    def _recon_loss(self, inputs: Dict) -> Tensor:
        """
        Computes the reconstruction loss component
        """

        x = inputs[GT_TENSOR_INPUTS_KEY]
        x_recon = inputs[OTHER_KEY]['reconstruction']
        loss = mse_loss(x, x_recon)

        return loss

    def _pred_loss(self, inputs: Dict) -> Tensor:
        """
        Computes the prediction loss component
        """

        y = inputs[GT_TENSOR_PREDICITONS_KEY]
        y_pred = inputs[MODELS_TENSOR_PREDICITONS_KEY]
        loss = mse_loss(y[:, :, :self._s_p], y_pred[:, :, :self._s_p])

        return loss

    def _lin_loss(self, inputs: Dict) -> Tensor:
        """
        Computes the linearity loss component
        """

        embedding_t = inputs[OTHER_KEY]['states_1'][:, :, :-1, :]
        embedding_t_plus_1 = inputs[OTHER_KEY]['states_0'][:, :, 1:, :]
        loss = mse_loss(embedding_t, embedding_t_plus_1)

        return loss

    def _inf_loss(self, inputs: Dict) -> Tensor:
        """
        Computes the infinity-norm loss component
        """

        x = inputs[GT_TENSOR_INPUTS_KEY]
        x_recon = inputs[OTHER_KEY]['reconstruction']
        recon_error = (x - x_recon).abs()
        recon_error_inf_norm = recon_error.view((recon_error.shape[0], -1)).max(dim=1)[
            0].mean()

        y = inputs[GT_TENSOR_PREDICITONS_KEY]
        y_pred = inputs[MODELS_TENSOR_PREDICITONS_KEY]
        pred_error_inf_norm = (y - y_pred).abs()
        pred_error_inf_norm = pred_error_inf_norm.view((pred_error_inf_norm.shape[0], -1)).max(dim=1)[
            0].mean()

        loss = recon_error_inf_norm + pred_error_inf_norm
        return loss

    def forward(self, inputs: Dict) -> Tensor:
        """
        The forward logic for UniversalLinearEmbeddingLoss.

        :param inputs: (Dict) Contains all of the required variables in order to compute all losses components.
        Specifically, it should contain t predictions, where t is the trajectory length chosen,
        the t Koopman operators used, the encoded states, and the reconstructions of each input.

        :return: (Tensor) The aggregated loss.
        """

        loss = (
                self._alpha_1 * (self._recon_loss(inputs) + self._pred_loss(inputs)) +
                self._lin_loss(inputs) +
                self._alpha_2 * self._inf_loss(inputs)
        )

        return loss


class ConsistentKoopmanLoss(LossComponent):
    """
    The combined loss criterion used in https://arxiv.org/pdf/2003.02236.pdf.
    """

    def __init__(
            self,
            k: int,
            lambda_id: float,
            lambda_fwd: float,
            lambda_bwd: float,
            lambda_con: float,
    ):
        """
        :param k: (int) Dynamics truncation dimensionality
        :param lambda_id: (float) Weight of the identification loss
        :param lambda_fwd: (float) Weight of the forward prediction loss
        :param lambda_bwd: (float) Weight of the backward prediction loss
        :param lambda_con: (float) Weight of the consistency loss
        """

        super(ConsistentKoopmanLoss, self).__init__()

        self._lambda_id = lambda_id
        self._lambda_fwd = lambda_fwd
        self._lambda_bwd = lambda_bwd
        self._lambda_con = lambda_con
        self._k = k

    def _id_loss(self, inputs: Dict) -> Tensor:
        """
        Computes the reconstruction loss component
        """

        x = inputs[GT_TENSOR_INPUTS_KEY]
        x_recon = inputs[OTHER_KEY]['reconstruction']
        loss = 0.5 * mse_loss(x, x_recon)

        return loss

    def _fwd_loss(self, inputs: Dict) -> Tensor:
        """
        Computes the prediction loss component
        """

        y = inputs[GT_TENSOR_PREDICITONS_KEY]
        y_pred = inputs[MODELS_TENSOR_PREDICITONS_KEY]
        loss = 0.5 * mse_loss(y, y_pred)

        return loss

    def _bwd_loss(self, inputs: Dict) -> Tensor:
        """
        Computes the linearity loss component
        """

        x = inputs[GT_TENSOR_INPUTS_KEY]
        x_pred = inputs[OTHER_KEY]['backward_pred']
        loss = 0.5 * mse_loss(x, x_pred)

        return loss

    def _con_loss(self, inputs: Dict) -> Tensor:
        """
        Computes the infinity-norm loss component
        """

        fwd_ops = inputs[OTHER_KEY]['D']
        bwd_ops = inputs[OTHER_KEY]['C']
        i_k = eye(self._k).to(fwd_ops[0].device)

        kappa = fwd_ops[0].shape[0] - self._k + 1
        consistencty_losses = [
            sum([
                (i_k - matmul(fwd_op[k:(k + self._k), :],
                              bwd_op[:, k:(k + self._k)])).norm().pow(2)
                for k in range(kappa)
            ])
            for fwd_op, bwd_op in zip(fwd_ops, bwd_ops)
        ]
        loss = (1 / (2 * kappa * len(fwd_ops))) * sum(consistencty_losses)

        return loss

    def forward(self, inputs: Dict) -> Tensor:
        """
        The forward logic for ConsistentKoopmanLoss.

        :param inputs: (Dict) Contains all of the required variables in order to compute all losses components.
        Specifically, it should contain t forward predictions, where t is the trajectory length chosen,
        t backward predictions, the reconstructions of each input and the t Koopman operators used.

        :return: (Tensor) The aggregated loss.
        """

        loss = (
                self._lambda_id * self._id_loss(inputs) +
                self._lambda_fwd * self._fwd_loss(inputs) +
                self._lambda_bwd * self._bwd_loss(inputs) +
                self._lambda_con * self._con_loss(inputs)
        )

        return loss


class MASELossModule(nn.Module):
    """
    A module implementing the Mean absolute scaled error metric
    """

    def __init__(self, m: int = 1, eps: float = 1e-8, accountable_observables: Optional[Sequence[int]] = None):
        """
        :param m: (int) Seasonality factor, if no-seasonality is assumed, m should be 1.
        :param eps: (float) Min value for the denominator in case it is equal to 0
        """

        super().__init__()

        self._m = m
        self._eps = eps

        if accountable_observables is not None:
            self._accountable_observables = torch.tensor(accountable_observables)[None, None, None, :]

        else:
            self._accountable_observables = None

    def forward(self, y: Tensor, y_pred: Tensor):
        """
        :param y: (Tensor) Ground-truth
        :param y_pred: (Tensor) Predictions

        :return: (Tensor) The MASE score
        """

        naive_predictions = (
                y[..., self._m::self._m, :] -
                y[..., :-self._m:self._m, :]
        )
        naive_predictions = naive_predictions.abs()
        predictions_errors = (y - y_pred).abs()

        if self._accountable_observables is not None:
            accountable_observables = self._accountable_observables.to(naive_predictions.device)

            naive_predictions = naive_predictions * accountable_observables
            predictions_errors = predictions_errors * accountable_observables

        naive_predictions = naive_predictions.mean()
        naive_predictions = torch.where(
            naive_predictions == 0,
            self._eps * torch.ones_like(naive_predictions),
            naive_predictions,
        )
        predictions_errors = predictions_errors.mean()
        mase = predictions_errors / naive_predictions

        return mase


class RMSELoss(nn.Module):
    """
    A module implementing the Mean absolute scaled error metric
    """

    def __init__(self):
        """
        :param m: (int) Seasonality factor, if no-seasonality is assumed, m should be 1.
        :param eps: (float) Min value for the denominator in case it is equal to 0
        """

        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, y: Tensor, y_pred: Tensor):
        loss = self.mse(y, y_pred).sqrt()

        return loss


class RelativeRMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y: Tensor, y_pred: Tensor):
        loss = ((y - y_pred).pow(2) / y.pow(2)).mean().sqrt()
        return loss


class MASELoss(LossComponent, MASELossModule):
    def __init__(
            self,
            m: int = 1,
            eps: float = 1e-8,
            trajectory_length: int = 1,
            accountable_observables: Optional[Sequence[int]] = None,
    ):
        LossComponent.__init__(
            self
        )

        MASELossModule.__init__(
            self,
            m=m,
            eps=eps,
            accountable_observables=accountable_observables
        )

        self._trajectory_length = trajectory_length

    def forward(self, inputs: Dict) -> Tensor:
        x = inputs[GT_TENSOR_INPUTS_KEY]
        y = inputs[GT_TENSOR_PREDICITONS_KEY]
        y_pred = inputs[MODELS_TENSOR_PREDICITONS_KEY]

        predictions_errors = (y - y_pred).abs()

        if y.shape[2] == self._m:
            naive_predictions = y - x[:, :, -1, :][:, :, None, :]

        else:
            naive_predictions = torch.cat(
                [
                    (y[:, :, (self._trajectory_length * m):(self._trajectory_length * (m + 1)), :] - x)
                    for m in range(self._m)
                ],
                dim=2,
            )

        naive_predictions = naive_predictions.abs()

        if self._accountable_observables is not None:
            accountable_observables = self._accountable_observables.to(naive_predictions.device)
            naive_predictions = naive_predictions * accountable_observables
            predictions_errors = predictions_errors * accountable_observables

        naive_predictions = naive_predictions.mean()
        naive_predictions = torch.where(
            naive_predictions == 0,
            self._eps * torch.ones_like(naive_predictions),
            naive_predictions,
        )
        predictions_errors = predictions_errors.mean()
        mase = predictions_errors / naive_predictions

        return mase
