# Implementation of the Consistent Koopman Auto-Encoder model, which was presented in:
# Forecasting Sequential Data Using Consistent Koopman Autoencoders: https://arxiv.org/pdf/2003.02236.pdf
# The code was taken from https://github.com/erichson/koopmanAE

from torch import nn, Tensor
from typing import Union, Dict

from midst.models.fc_models import FCEncoderDecoder
from midst.utils.defaults import MODELS_TENSOR_PREDICITONS_KEY, OTHER_KEY

import torch


def gaussian_init_(n_units, std=1):
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std / n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]
    return Omega


class dynamics(nn.Module):
    def __init__(self, b, init_scale):
        super(dynamics, self).__init__()
        self.dynamics = nn.Linear(b, b, bias=False)
        self.dynamics.weight.data = gaussian_init_(b, std=1)
        U, _, V = torch.svd(self.dynamics.weight.data)
        self.dynamics.weight.data = torch.mm(U, V.t()) * init_scale

    def forward(self, x):
        x = self.dynamics(x)
        return x


class dynamics_back(nn.Module):
    def __init__(self, b, omega):
        super(dynamics_back, self).__init__()
        self.dynamics = nn.Linear(b, b, bias=False)
        self.dynamics.weight.data = torch.pinverse(omega.dynamics.weight.data.t())

    def forward(self, x):
        x = self.dynamics(x)
        return x


class CK(nn.Module):
    def __init__(
            self,
            states_dim: int,
            observable_dim: int,
            m_dynamics: int = 1,
            n_layers_encoder: int = 8,
            l0_units: int = 1024,
            units_factor: float = 0.5,
            activation: Union[str, dict] = 'relu',
            final_activation: Union[str, dict] = None,
            norm: str = None,
            dropout: float = None,
            bias: bool = False,
            k_prediction_steps: int = 1,
            steps: int = 8,
            steps_back: int = 8,
            init_scale: float = 1,
            simple_dynamics: bool = False,
    ):
        super(CK, self).__init__()
        self.steps = steps
        self.steps_back = steps_back
        self.k_prediction_steps = k_prediction_steps
        self.simple_dynamics = simple_dynamics
        self._m_dynamics = m_dynamics
        self._states_dim = states_dim
        self._observable_dim = observable_dim

        self.encoder = FCEncoderDecoder(
            input_dim=states_dim,
            output_dim=observable_dim,
            n_layers=n_layers_encoder,
            l0_units=l0_units,
            units_factor=units_factor,
            activation=activation,
            final_activation=final_activation,
            norm=norm,
            dropout=dropout,
            bias=bias,
        )
        self.decoder = FCEncoderDecoder(
            input_dim=observable_dim,
            output_dim=states_dim,
            n_layers=n_layers_encoder,
            l0_units=int(l0_units * (units_factor ** (n_layers_encoder - 1))),
            units_factor=(1 / units_factor),
            activation=activation,
            final_activation=final_activation,
            norm=norm,
            dropout=dropout,
            bias=bias,
        )

        if simple_dynamics:
            self.dynamics = nn.ParameterList(
                [
                    nn.parameter.Parameter(
                        nn.init.xavier_normal_(
                            torch.zeros(
                                (observable_dim, observable_dim)
                            )
                        ).type(torch.float32), requires_grad=True,
                    )
                    for _ in range(k_prediction_steps)
                    for _ in range(m_dynamics)
                ]
            )

            self.backdynamics = nn.ParameterList(
                [
                    nn.parameter.Parameter(
                        nn.init.xavier_normal_(
                            torch.zeros(
                                (observable_dim, observable_dim)
                            )
                        ).type(torch.float32), requires_grad=True,
                    )
                    for _ in range(k_prediction_steps)
                    for _ in range(m_dynamics)
                ]
            )

        else:
            self.dynamics = nn.ModuleList(
                [
                    dynamics(observable_dim, init_scale)
                    for _ in range(k_prediction_steps)
                    for _ in range(m_dynamics)
                ]
            )
            self.backdynamics = nn.ModuleList(
                [
                    dynamics_back(observable_dim, self.dynamics[i])
                    for i in range(k_prediction_steps)
                    for _ in range(m_dynamics)
                ]
            )

    def forward(self, x: torch.Tensor) -> Dict:
        # Encode
        z_fwd = self.encoder(x)
        z_bwd = self.encoder(x.flip(dims=(2,)))

        # Predict
        if self.simple_dynamics:
            reshaped_z = z_fwd.reshape((z_fwd.shape[0] * z_fwd.shape[1]), z_fwd.shape[2], z_fwd.shape[3])
            koopmans = [
                torch.cat(
                    [
                        d[None, ...]
                        for d in self.dynamics[(j * self._m_dynamics):((j + 1) * self._m_dynamics)]
                    ],
                    dim=0
                )
                for j in range(self.k_prediction_steps)
            ]
            q_fwd = torch.cat(
                [
                    torch.reshape(
                        torch.bmm(
                            koopmans_t.repeat(z_fwd.shape[0], 1, 1),
                            reshaped_z.transpose(1, 2)
                        ).transpose(1, 2),
                        z_fwd.shape
                    )
                    for koopmans_t in koopmans
                ],
                dim=2
            )

            reshaped_z = z_bwd.reshape((z_bwd.shape[0] * z_bwd.shape[1]), z_bwd.shape[2], z_bwd.shape[3])
            koopmans = [
                torch.cat(
                    [
                        d[None, ...]
                        for d in self.backdynamics[(j * self._m_dynamics):((j + 1) * self._m_dynamics)]
                    ],
                    dim=0
                )
                for j in range(self.k_prediction_steps)
            ]
            q_bwd = torch.cat(
                [
                    torch.reshape(
                        torch.bmm(
                            koopmans_t.repeat(z_bwd.shape[0], 1, 1),
                            reshaped_z.transpose(1, 2)
                        ).transpose(1, 2),
                        z_bwd.shape
                    )
                    for koopmans_t in koopmans
                ],
                dim=2
            )

        else:
            q_fwd = [
                [
                    dyn(z_fwd[:, i, ...])[:, None, ...]
                    for i, dyn in enumerate(self.dynamics[(j * self._m_dynamics):((j + 1) * self._m_dynamics)])
                ]
                for j in range(self.k_prediction_steps)
            ]
            q_fwd = torch.cat(
                [
                    torch.cat(dyn, dim=1)
                    for dyn in q_fwd
                ],
                dim=2,
            )
            q_bwd = [
                [
                    dyn(z_bwd)[:, i, ...][:, None, ...]
                    for i, dyn in enumerate(self.backdynamics[(j * self._m_dynamics):((j + 1) * self._m_dynamics)])
                ]
                for j in range(self.k_prediction_steps)
            ]
            q_bwd = torch.cat(
                [
                    torch.cat(dyn, dim=1)
                    for dyn in q_bwd
                ],
                dim=2,
            )

        fwd_predictions = self.decoder(q_fwd)
        bwd_predictions = self.decoder(q_bwd)
        bwd_predictions = (
            bwd_predictions[..., :x.shape[2], :]
            if self.k_prediction_steps > 1
            else bwd_predictions
        )

        # Reconstruct
        reconstruction = self.decoder(z_fwd)

        out = {
            MODELS_TENSOR_PREDICITONS_KEY: fwd_predictions,
            OTHER_KEY: {
                'D': self.dynamics if self.simple_dynamics else [dyn.dynamics.weight for dyn in self.dynamics],
                'C': self.backdynamics if self.simple_dynamics else [dyn.dynamics.weight for dyn in self.backdynamics],
                'reconstruction': reconstruction,
                'backward_pred': bwd_predictions,
                'dynamics': self.dynamics if self.simple_dynamics else [dyn.dynamics.weight for dyn in self.dynamics],
            }
        }

        return out
