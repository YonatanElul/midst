# Implementation of the Koopman Auto-Encoder model, which was presented in:
# Deep learning for universal linear embeddings of nonlinear dynamics:
# https://www.nature.com/articles/s41467-018-07210-0.pdf
# This PyTorch implementation is also based on the authors' TensorFlow implementation which is available at:
# https://github.com/BethanyL/DeepKoopman

from torch import Tensor
from typing import Union, Dict
from midst.models.fc_models import FCEncoderDecoder, init_weights
from midst.utils.defaults import (
    MODELS_TENSOR_PREDICITONS_KEY,
    OTHER_KEY
)

import torch
import torch.nn as nn


class UniversalLinearEmbeddingAE(nn.Module):
    """
    A PyTorch-based implementation for the Koopman Auto-Encoder presented in:
    https://www.nature.com/articles/s41467-018-07210-0.pdf
    """

    def __init__(
            self,
            t: int,
            input_dim: int,
            n_layers_encoder: int = 8,
            n_layers_decoder: int = 8,
            n_layers_aux: int = 8,
            l0_units: int = 1024,
            l0_units_aux: int = 1024,
            units_factor: float = 0.5,
            units_factor_aux: int = 2.,
            activation: Union[str, dict] = 'relu',
            final_activation: Union[str, dict] = None,
            norm: str = None,
            dropout: float = None,
            bias: bool = False,
            k_prediction_steps: int = 1,
    ):
        """
        The constructor of the FCAE class, which is a composition of to symmetric FCEncoderDecoder models, one
        serving as an encoder and one as a decoder.

        :param t: (int) Number of time-steps for which the model should make prediction. For each time-step we need to
        generate a new Koopman operator, parametrized by different eigenvalues.
        :param input_dim: (int) Dimensionality of the inputs.
        :param n_layers_encoder: (int) Number of FC layers to include in the encoder.
        :param n_layers_encoder: (int) Number of FC layers to include in the decoder.
        :param n_layers_aux: (int) Number of FC layers to include the aux network.
        :param l0_units_aux: (int) Number units to use in the first layer of the aux network.
        :param units_factor_aux: (float) Multiplicative factor for increasing the number of units in
        each consecutive FC layer in the aux network.
        :param k_prediction_steps: (int) How many prediction steps into the future to predict, generates a different
        Koopman operator for each step.
        """

        super().__init__()

        self._t = t
        self._encoder = FCEncoderDecoder(
            input_dim=input_dim,
            output_dim=2,
            n_layers=n_layers_encoder,
            l0_units=l0_units,
            units_factor=units_factor,
            activation=activation,
            final_activation=final_activation,
            norm=norm,
            dropout=dropout,
            bias=bias,
        )

        self._aux_models = nn.ModuleList([
            FCEncoderDecoder(
                input_dim=2,
                output_dim=2,
                n_layers=n_layers_aux,
                l0_units=l0_units_aux,
                units_factor=units_factor_aux,
                activation=activation,
                final_activation=final_activation,
                norm=norm,
                dropout=dropout,
                bias=bias,
            ) for _ in range(k_prediction_steps)
        ])

        self._decoder = FCEncoderDecoder(
            input_dim=2,
            output_dim=input_dim,
            n_layers=n_layers_decoder,
            l0_units=int(l0_units * (units_factor ** (n_layers_encoder - 1))),
            units_factor=(1 / units_factor),
            activation=activation,
            final_activation=final_activation,
            norm=norm,
            dropout=dropout,
            bias=bias,
        )

        # Initialize weights
        self._encoder.apply(init_weights)
        self._decoder.apply(init_weights)
        [aux.apply(init_weights) for aux in self._aux_models]
        self._k_prediction_steps = k_prediction_steps

    def forward(self, x: Tensor) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
        """
        The forward logic for the 'UniversalLinearEmbeddingAE' class.

        :param x: (Tensor) The input tensor.

        :return: (Tensor) The resulting tensor from the forward pass.
        """

        # Validate inputs
        assert len(x.shape) == 4, \
            f"The input tensor x must be a 4-dimensional Tensor, however x is " \
            f"a {len(x.shape)}-dimensional Tensor."

        # Encode each row of observables
        embeddings = self._encoder(x)

        # Split the different dynamics in order to decouple their dynamics
        embeddings = embeddings.split(1, dim=1)

        # Multiply at each time step by the appropriate Koopman matrix
        lambdas = [
            [aux(e.squeeze()) for e in embeddings]
            for aux in self._aux_models
        ]

        # Expand if there is just one system
        if len(lambdas[0][0].shape) < 3:
            lambdas = [
                [
                    lambda_[None, ...]
                    for lambda_ in aux_lambdas
                ]
            for aux_lambdas in lambdas
            ]

        first_cols = [
            [
                torch.cat([
                    lambda_[:, :, 1].unsqueeze(-1).cos(), lambda_[:, :, 1].unsqueeze(-1).sin()
                ], dim=2).unsqueeze(-1)
                for lambda_ in aux_lambdas
            ]
            for aux_lambdas in lambdas
        ]
        second_cols = [
            [
                torch.cat([
                    -lambda_[:, :, 1].unsqueeze(-1).sin(), lambda_[:, :, 1].unsqueeze(-1).cos()
                ], dim=2).unsqueeze(-1)
                for lambda_ in aux_lambdas
            ]
            for aux_lambdas in lambdas
        ]
        koopmans_t = [
            [torch.cat([cols_1[k], cols_2[k]], dim=-1) for k in range(x.shape[1])]
            for cols_1, cols_2 in zip(first_cols, second_cols)
        ]
        koopmans_t = [
            [
                (
                        torch.ones_like(koopmans_per_t[k]) * lambdas[t][k][:, :, 0, None, None]
                ).exp() * koopmans_per_t[k]
                for k in range(x.shape[1])
            ]
            for t, koopmans_per_t in enumerate(koopmans_t)
        ]
        next_states = [
            [
                torch.bmm(
                    koopmans_per_t[k].view(x.shape[0] * x.shape[2], 2, 2),
                    embeddings[k].squeeze(1).reshape(x.shape[0] * x.shape[2], 1, 2).transpose(2, 1)
                ).transpose(2, 1).view(x.shape[0], x.shape[2], 2).unsqueeze(1)
                for k in range(x.shape[1])
            ]
            for t, koopmans_per_t in enumerate(koopmans_t)
        ]
        next_states = torch.cat([torch.cat(states, 1) for states in next_states], 2)
        embeddings = torch.cat(embeddings, 1)

        # Decode each row of observables
        reconstruction = self._decoder(embeddings)
        predictions = self._decoder(next_states)

        output = {
            MODELS_TENSOR_PREDICITONS_KEY: predictions,
            OTHER_KEY: {
                'states_0': embeddings,
                'states_1': next_states[..., :embeddings.shape[2], :],
                'reconstruction': reconstruction,
                'dynamics': koopmans_t,
            },
        }

        return output
