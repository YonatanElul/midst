from abc import abstractmethod
from scipy.stats import ortho_group
from torch import Tensor, cat, diag, bmm
from typing import Union, Dict, Sequence, Tuple
from midst.models.fc_models import FCEncoderDecoder
from midst.utils.defaults import MODELS_TENSOR_PREDICITONS_KEY, OTHER_KEY

import torch
import torch.nn as nn


class BaseMIDST(nn.Module):
    def __init__(
            self,
            m_dynamics: int,
            observable_dim: int = 64,
            min_init_eig_val: float = 0.01,
            max_init_eig_val: float = 1.,
            k_forward_prediction: int = 1,
            eps: float = 1e-4,
            identical_sigmas: bool = False,
            symmetric: bool = False,
            residual_dynamics: bool = False,
            dmf_dynamics: bool = False,
            separate_dynamics: bool = False,
            spectral_norm: bool = False,
            temporal_sharing: bool = False,
    ):
        super(BaseMIDST, self).__init__()

        self._m_dynamics = m_dynamics
        self._observable_dim = observable_dim
        self._k_forward_prediction = k_forward_prediction
        self._eps = eps
        self._identical_sigmas = identical_sigmas
        self._symmetric = symmetric
        self._residual_dynamics = residual_dynamics
        self._dmf_dynamics = dmf_dynamics
        self._separate_dynamics = separate_dynamics
        self._spectral_norm = spectral_norm
        self._temporal_sharing = temporal_sharing

        # Initialize dynamics
        if dmf_dynamics:
            if separate_dynamics:
                U = [
                    [
                        nn.init.xavier_normal_(
                            torch.zeros(
                                (observable_dim, observable_dim)
                            )
                        ).type(torch.float32)
                        for _ in range(m_dynamics)
                    ]
                    for _ in range(k_forward_prediction)
                ]
                self._U_per_t = nn.ModuleList(
                    [
                        nn.ParameterList(
                            [
                                nn.parameter.Parameter(
                                    u, requires_grad=True,
                                )
                                for u in u_per_t
                            ]
                        )
                        for u_per_t in U
                    ]
                )
                V = [
                    [
                        nn.init.xavier_normal_(
                            torch.zeros(
                                (observable_dim, observable_dim)
                            )
                        ).type(torch.float32)
                        for _ in range(m_dynamics)
                    ]
                    for _ in range(k_forward_prediction)
                ]
                self._V_per_t = nn.ModuleList(
                    [
                        nn.ParameterList(
                            [
                                nn.parameter.Parameter(
                                    v, requires_grad=True,
                                )
                                for v in v_per_t
                            ]
                        )
                        for v_per_t in V
                    ]
                )

            else:
                U = [
                    nn.init.xavier_normal_(
                        torch.zeros(
                            (observable_dim, observable_dim)
                        )
                    ).type(torch.float32)
                    for _ in range(k_forward_prediction)
                ]
                self._U_per_t = nn.ParameterList(
                    [
                        nn.parameter.Parameter(
                            u, requires_grad=True,
                        )
                        for u in U
                    ]
                )
                V = [
                    nn.init.xavier_normal_(
                        torch.zeros(
                            (observable_dim, observable_dim)
                        )
                    ).type(torch.float32)
                    for _ in range(k_forward_prediction)
                ]
                self._V_per_t = nn.ParameterList(
                    [
                        nn.parameter.Parameter(
                            v, requires_grad=True,
                        )
                        for v in V
                    ]
                )

            if identical_sigmas:
                S = [
                    [
                        nn.init.xavier_normal_(
                            torch.zeros(
                                (observable_dim, observable_dim)
                            )
                        ).type(torch.float32)
                        for _ in range(m_dynamics)
                    ]
                    for _ in range(k_forward_prediction)
                ]
                self._S_per_t_per_m = nn.ModuleList(
                    [
                        nn.ParameterList(
                            [
                                nn.parameter.Parameter(
                                    S[t][0], requires_grad=True,
                                )
                            ]
                        )
                        for t in range(k_forward_prediction)
                    ]
                )

            else:
                S = [
                    [
                        nn.init.xavier_normal_(
                            torch.zeros(
                                (observable_dim, observable_dim)
                            )
                        ).type(torch.float32)
                        for _ in range(m_dynamics)
                    ]
                    for _ in range(k_forward_prediction)
                ]
                self._S_per_t_per_m = nn.ModuleList(
                    [
                        nn.ParameterList(
                            [
                                nn.parameter.Parameter(
                                    S[t][m], requires_grad=True,
                                )
                                for m in range(m_dynamics)
                            ]
                        )
                        for t in range(k_forward_prediction)
                    ]
                )

        else:
            if separate_dynamics:
                U = [
                    [
                        torch.from_numpy(
                            ortho_group.rvs(observable_dim)
                        ).type(torch.float32)
                        for _ in range(m_dynamics)
                    ]
                    for _ in range(k_forward_prediction)
                ]
                self._U_per_t = nn.ModuleList(
                    [
                        nn.ParameterList(
                            [
                                nn.parameter.Parameter(
                                    u, requires_grad=True,
                                )
                                for u in u_per_t
                            ]
                        )
                        for u_per_t in U
                    ]
                )
                V = [
                    [
                        torch.from_numpy(
                            ortho_group.rvs(observable_dim)
                        ).type(torch.float32)
                        for _ in range(m_dynamics)
                    ]
                    for _ in range(k_forward_prediction)
                ]
                self._V_per_t = nn.ModuleList(
                    [
                        nn.ParameterList(
                            [
                                nn.parameter.Parameter(
                                    v, requires_grad=True,
                                )
                                for v in v_per_t
                            ]
                        )
                        for v_per_t in V
                    ]
                )

            else:
                U = [
                    torch.from_numpy(
                        ortho_group.rvs(observable_dim)
                    ).type(torch.float32)
                    for _ in range((k_forward_prediction if not temporal_sharing else 1))
                ]
                self._U_per_t = nn.ParameterList(
                    [
                        nn.parameter.Parameter(
                            u, requires_grad=True,
                        )
                        for u in U
                    ]
                )
                V = [
                    torch.from_numpy(
                        ortho_group.rvs(observable_dim)
                    ).type(torch.float32)
                    for _ in range((k_forward_prediction if not temporal_sharing else 1))
                ]
                self._V_per_t = nn.ParameterList(
                    [
                        nn.parameter.Parameter(
                            v, requires_grad=True,
                        )
                        for v in V
                    ]
                )

            if max_init_eig_val is None or min_init_eig_val is None:
                sigmas = [
                    [
                        torch.normal(
                            torch.zeros((observable_dim,)),
                            0.5 * torch.ones((observable_dim,)),
                        ).abs()
                        for _ in range(m_dynamics)
                    ]
                    for _ in range(k_forward_prediction)
                ]

            else:
                sigmas = [
                    [
                        torch.linspace(
                            start=max_init_eig_val,
                            end=min_init_eig_val,
                            steps=observable_dim,
                        ).abs()
                        for _ in range(m_dynamics)
                    ]
                    for _ in range(k_forward_prediction)
                ]

            if identical_sigmas:
                self._S_per_t_per_m = torch.nn.ParameterList(
                    [
                        nn.parameter.Parameter(
                            sigmas_per_t[0], requires_grad=True,
                        )
                        for sigmas_per_t in sigmas
                    ]
                )

            else:
                self._S_per_t_per_m = torch.nn.ModuleList(
                    [
                        nn.ParameterList(
                            [
                                nn.parameter.Parameter(
                                    sigma, requires_grad=True,
                                )
                                for sigma in sigmas_per_t
                            ]
                        )
                        for sigmas_per_t in sigmas
                    ]
                )

    def _get_sigmas(self) -> Union[nn.ModuleList, nn.ParameterList]:
        if self._identical_sigmas:
            if self._dmf_dynamics:
                sigmas = tuple(
                    tuple(
                        self._S_per_t_per_m[t]
                        for _ in range(self._m_dynamics)
                    )
                    for t in range(self._k_forward_prediction)
                )

            else:
                sigmas = tuple(
                    tuple(
                        diag(self._S_per_t_per_m[t].abs().sort(descending=True)[0])
                        for _ in range(self._m_dynamics)
                    )
                    for t in range(self._k_forward_prediction)
                )

        else:
            if self._dmf_dynamics:
                sigmas = self._S_per_t_per_m

            else:
                sigmas = tuple(
                    tuple(
                        diag(self._S_per_t_per_m[t][m].abs().sort(descending=True)[0])
                        for m in range(self._m_dynamics)
                    )
                    for t in range(self._k_forward_prediction)
                )

        return sigmas

    def _get_u(self) -> Union[Sequence[Tensor], nn.ModuleList, nn.ParameterList]:
        return self._U_per_t

    def _get_v(self) -> Union[Sequence[Tensor], nn.ModuleList, nn.ParameterList]:
        if self._symmetric and not self._dmf_dynamics:
            return self._U_per_t

        else:
            return self._V_per_t

    def _get_koopmans(
            self,
    ) -> Sequence[Tensor]:
        s = self._get_sigmas()
        u = self._get_u()
        v = self._get_v()

        if self._separate_dynamics:
            koopmans = [
                torch.cat(
                    [
                        (u[t][m].matmul(s[t][m]).matmul(v[t][m].transpose(1, 0)))[None, ...]
                        for m in range(self._m_dynamics)
                    ],
                    dim=0
                )
                for t in range(self._k_forward_prediction)
            ]

        else:
            koopmans = [
                torch.cat(
                    [
                        (
                            u[0 if self._temporal_sharing else t].matmul(s[t][m]).matmul(
                                v[0 if self._temporal_sharing else t].transpose(1, 0)
                            )
                        )[None, ...]
                        for m in range(self._m_dynamics)
                    ],
                    dim=0
                )
                for t in range(self._k_forward_prediction)
            ]

        return koopmans

    def _apply_koopmans(self, observables: Tensor) -> Tuple[Tensor, Sequence[Sequence[Tensor]]]:
        # If in residual mode than first compute the residuals
        if self._residual_dynamics:
            assert observables.shape[2] > 1, f"Cannot use residual mode with {observables.shape[2]=}"
            x = observables[..., 1:, :] - observables[..., :-1, :]

        else:
            x = observables

        # Unpack to each temporal element of each dynamic component in the batch
        observables_ = torch.reshape(
            x,
            ((x.shape[0] * x.shape[1]), x.shape[2], x.shape[3])
        )

        # Apply the evolution operators
        koopmans_per_system_per_t = self._get_koopmans()
        observables_per_system_per_t = cat(
            [
                torch.reshape(
                    bmm(
                        koopmans_t.repeat(observables.shape[0], 1, 1),
                        observables_.transpose(1, 2)
                    ).transpose(1, 2),
                    x.shape
                )
                for koopmans_t in koopmans_per_system_per_t
            ],
            dim=2
        )

        # If in residual mode than add back the initial values
        if self._residual_dynamics:
            if self._k_forward_prediction > 1:
                observables_per_system_per_t = torch.split(
                    (
                            observables[..., 0, :][..., None, :] + observables_per_system_per_t.cumsum(dim=2)
                    ),
                    x.shape[2],
                    dim=2,
                )
                observables_per_system_per_t = [
                    torch.cat([observables[..., 0, :][..., None, :], obs], dim=2)
                    for obs in observables_per_system_per_t
                ]
                observables_per_system_per_t = torch.cat(observables_per_system_per_t, dim=2)

            else:
                observables_per_system_per_t = cat(
                    (
                        observables[..., 0, :][..., None, :],
                        observables[..., 0, :][..., None, :] + observables_per_system_per_t.cumsum(dim=2),
                    ),
                    dim=2
                )

        return observables_per_system_per_t, koopmans_per_system_per_t

    def singular_vectors_params(self) -> Sequence[torch.nn.Parameter]:
        params = self._U_per_t

        for v in self._V_per_t:
            params += v

        return params

    def singular_values_params(self) -> Sequence[torch.nn.Parameter]:
        params = []
        for p in self._S_per_t_per_m:
            params += p

        return params

    @abstractmethod
    def forward(self, x: Tensor) -> Dict:
        pass


class MIDST(BaseMIDST):
    def __init__(
            self,
            m_dynamics: int,
            states_dim: int,
            observable_dim: int = 64,
            n_encoder_layers: int = 8,
            l0_units: int = 1024,
            units_factor: float = 0.5,
            activation: Union[str, dict] = 'relu',
            final_activation: Union[str, dict] = None,
            norm: Union[dict, str] = None,
            dropout: float = 0.2,
            bias: bool = False,
            min_init_eig_val: float = 0.01,
            max_init_eig_val: float = 1.,
            k_forward_prediction: int = 1,
            eps: float = 1e-4,
            identical_sigmas: bool = False,
            symmetric: bool = False,
            residual_dynamics: bool = False,
            dmf_dynamics: bool = False,
            separate_dynamics: bool = False,
            spectral_norm: bool = False,
            n_forwards: int = 1,
            temporal_sharing: bool = True,
            mc_dropout: bool = True,
            skip_connections: bool = False,
    ):
        super(MIDST, self).__init__(
            m_dynamics=m_dynamics,
            observable_dim=observable_dim,
            min_init_eig_val=min_init_eig_val,
            max_init_eig_val=max_init_eig_val,
            k_forward_prediction=k_forward_prediction,
            eps=eps,
            identical_sigmas=identical_sigmas,
            symmetric=symmetric,
            residual_dynamics=residual_dynamics,
            dmf_dynamics=dmf_dynamics,
            separate_dynamics=separate_dynamics,
            spectral_norm=spectral_norm,
            temporal_sharing=temporal_sharing,
        )

        self._states_dim = states_dim
        self._n_forwards = n_forwards
        self._dropout = dropout

        assert int(separate_dynamics and identical_sigmas) == 0

        if n_encoder_layers > 0:
            self.encoder = FCEncoderDecoder(
                input_dim=states_dim,
                output_dim=observable_dim,
                n_layers=n_encoder_layers,
                l0_units=l0_units,
                units_factor=units_factor,
                activation=activation,
                final_activation=final_activation,
                norm=norm,
                dropout=dropout,
                bias=bias,
                spectral_norm=spectral_norm,
                mc_dropout=mc_dropout,
                skip_connections=skip_connections,
            )
            self.decoder = FCEncoderDecoder(
                input_dim=observable_dim,
                output_dim=states_dim,
                n_layers=n_encoder_layers,
                l0_units=int(l0_units * (units_factor ** (n_encoder_layers - 1))),
                units_factor=(1 / units_factor),
                activation=activation,
                final_activation=final_activation,
                norm=norm,
                dropout=dropout,
                bias=bias,
                spectral_norm=spectral_norm,
                mc_dropout=mc_dropout,
                skip_connections=skip_connections,
            )

    def forward(self, states_t_0: Tensor) -> Dict[str, Tensor]:
        # Validate inputs
        assert len(states_t_0.shape) == 4, \
            f"The input tensor x must be a 4-dimensional Tensor, " \
            f"however states_t_0 is a " \
            f"{len(states_t_0.shape)}-dimensional Tensor."

        assert states_t_0.shape[1] == self._m_dynamics, \
            f"The model expects {self._m_dynamics} " \
            f"dynamics, however the input tensor states_t_0 has only " \
            f"{states_t_0.shape[1]} dynamics."

        assert states_t_0.shape[3] == self._states_dim, \
            f"The model expects {self._states_dim} observable components," \
            f" however the input tensor states_t_0 has {states_t_0.shape[3]} " \
            f"observable components."

        # Encode each row of observables
        if self._n_forwards > 1:
            scaling_factor = (
                1 if not self._dropout else (1 / (1 - self._dropout))
            )
            observables_t_0 = [
                self.encoder(states_t_0)[None, ...] * scaling_factor
                for _ in range(self._n_forwards)
            ]
            observables_t_0 = torch.cat(observables_t_0, dim=0).mean(0)

        else:
            observables_t_0 = self.encoder(states_t_0)

        # Apply the forward & inverse Koopman operators
        states_1, dynamics = self._apply_koopmans(observables_t_0)

        # Decode & reshape back to original shape
        states_t = self.decoder(states_1)
        reconstruction = self.decoder(observables_t_0)

        output = {
            MODELS_TENSOR_PREDICITONS_KEY: states_t,
            OTHER_KEY: {
                'states_0': observables_t_0,
                'states_1': states_1,
                'reconstruction': reconstruction,
                'dynamics': dynamics,
                'V_per_t': self._V_per_t,
                'U_per_t': self._U_per_t,
                'S_per_t': self._S_per_t_per_m,
            },
        }

        return output

    def nn_params(self) -> Sequence[torch.nn.Parameter]:
        params = list(self.encoder.parameters())
        params += list(self.decoder.parameters())
        return params
