from abc import ABC
from typing import Sequence, List, Union, Optional

import numpy as np


class StrangeAttractor(ABC):
    """
    A class which manages the generation of chaotic attractors.
    """

    DISTS_TYPES = (
        'normal',
        'uniform',
    )

    def __init__(
            self,
            attractor_type: str,
            x0: np.ndarray,
            attractor_params: Optional[dict] = None,
            noise_dist: dict = None,
    ):
        super().__init__()

        self.ATTRACTORS = {
            'lorenz': self._lorenz_step,
            'thomas': self._thomas_step,
            'chen': self._chen_step,
            'halvorsen': self._halvorsen_step,
            'sprott': self._sprott_step,
            'three-scrolls': self._three_scrolls_step,
            'four-wings': self._four_wings_step,
        }

        assert attractor_type in self.ATTRACTORS

        if attractor_type == 'lorenz':
            if attractor_params is not None:
                assert len(attractor_params) == 3
                assert [p in attractor_params for p in ('sigma', 'rho', 'beta')]

            else:
                attractor_params = {
                    'sigma': 10,
                    'rho': 28,
                    'beta': 8 / 3,
                }

        elif attractor_type == 'thomas':
            if attractor_params is not None:
                assert len(attractor_params) == 1
                assert 'b' in attractor_params

            else:
                attractor_params = {
                    'b': 0.208186
                }

        elif attractor_type == 'chen':
            if attractor_params is not None:
                assert len(attractor_params) == 3
                assert [p in attractor_params for p in ('alpha', 'beta', 'delta')]

            else:
                attractor_params = {
                    'alpha': 5,
                    'beta': -10,
                    'delta': -0.38,
                }

        elif attractor_type == 'halvorsen':
            if attractor_params is not None:
                assert len(attractor_params) == 1
                assert 'a' in attractor_params

            else:
                attractor_params = {
                    'a': 1.89
                }

        elif attractor_type == 'sprott':
            if attractor_params is not None:
                assert len(attractor_params) == 2
                assert [p in attractor_params for p in ('a', 'b')]

            else:
                attractor_params = {
                    'a': 2.07,
                    'b': 1.79,
                }

        elif attractor_type == 'three-scrolls':
            if attractor_params is not None:
                assert len(attractor_params) == 6
                assert [p in attractor_params for p in ('a', 'b', 'c', 'd', 'e', 'f')]

            else:
                attractor_params = {
                    'a': 32.48,
                    'b': 45.84,
                    'c': 1.18,
                    'd': 0.13,
                    'e': 0.57,
                    'f': 14.7,
                }

        elif attractor_type == 'four-wings':
            if attractor_params is not None:
                assert len(attractor_params) == 3
                assert [p in attractor_params for p in ('a', 'b', 'c')]

            else:
                attractor_params = {
                    'a': 0.2,
                    'b': 0.01,
                    'c': -0.4,
                }

        if noise_dist is None:
            noise_dist = {
                'type': 'uniform',
                'low': -1,
                'high': 1,
            }

        else:
            assert noise_dist['type'] in self.DISTS_TYPES, \
                f"{noise_dist['type']} is not a valid distribution type. Please use on of " \
                f"{self.DISTS_TYPES}."

        self._attractor_type = attractor_type
        self._attractor_params = attractor_params
        self._x0 = x0
        self._noise_dist = noise_dist

    def next_state(self, states_0: np.ndarray, dt: float) -> np.ndarray:
        return self.ATTRACTORS[self._attractor_type](states_0=states_0, dt=dt)

    def _lorenz_step(self, states_0: np.ndarray, dt: float) -> np.ndarray:
        dx = self._attractor_params['sigma'] * (-states_0[0] + states_0[1])
        dy = -(states_0[0] * states_0[2]) + (self._attractor_params['rho'] * states_0[0]) - states_0[1]
        dz = (states_0[0] * states_0[1]) - (self._attractor_params['beta'] * states_0[2])

        states_1 = states_0 + np.array([dx, dy, dz]) * dt
        return states_1

    def _thomas_step(self, states_0: np.ndarray, dt: float) -> np.ndarray:
        dx = np.sin(states_0[1]) - (self._attractor_params['b'] * states_0[0])
        dy = np.sin(states_0[2]) - (self._attractor_params['b'] * states_0[1])
        dz = np.sin(states_0[0]) - (self._attractor_params['b'] * states_0[2])

        states_1 = states_0 + np.array([dx, dy, dz]) * dt
        return states_1

    def _chen_step(self, states_0: np.ndarray, dt: float) -> np.ndarray:
        dx = (self._attractor_params['alpha'] * states_0[0]) - (states_0[1] * states_0[2])
        dy = (self._attractor_params['beta'] * states_0[1]) + (states_0[0] * states_0[2])
        dz = (self._attractor_params['delta'] * states_0[2]) + (states_0[0] * states_0[1] / 3)

        states_1 = states_0 + np.array([dx, dy, dz]) * dt
        return states_1

    def _halvorsen_step(self, states_0: np.ndarray, dt: float) -> np.ndarray:
        dx = (
                (-self._attractor_params['a'] * states_0[0]) -
                (4 * states_0[1]) -
                (4 * states_0[2]) -
                np.power(states_0[1], 2)
        )
        dy = (
                (-self._attractor_params['a'] * states_0[1]) -
                (4 * states_0[2]) -
                (4 * states_0[0]) -
                np.power(states_0[2], 2)
        )
        dz = (
                (-self._attractor_params['a'] * states_0[2]) -
                (4 * states_0[0]) -
                (4 * states_0[1]) -
                np.power(states_0[0], 2)
        )

        states_1 = states_0 + np.array([dx, dy, dz]) * dt
        return states_1

    def _sprott_step(self, states_0: np.ndarray, dt: float) -> np.ndarray:
        dx = states_0[1] + (self._attractor_params['a'] * states_0[0] * states_0[1]) + (states_0[0] * states_0[2])
        dy = 1 - (self._attractor_params['b'] * np.power(states_0[0], 2)) + (states_0[1] * states_0[2])
        dz = states_0[0] - np.power(states_0[0], 2) - np.power(states_0[1], 2)

        states_1 = states_0 + np.array([dx, dy, dz]) * dt
        return states_1

    def _three_scrolls_step(self, states_0: np.ndarray, dt: float) -> np.ndarray:
        dx = (
                (self._attractor_params['a'] * (states_0[1] - states_0[0])) +
                (self._attractor_params['d'] * states_0[0] * states_0[1])
        )
        dy = (
                (self._attractor_params['b'] * states_0[0]) -
                (states_0[0] * states_0[1]) +
                (self._attractor_params['f'] * states_0[1])
        )
        dz = (
                (self._attractor_params['c'] * states_0[2]) +
                (states_0[0] * states_0[1]) -
                (self._attractor_params['e'] * np.power(states_0[0], 2))
        )

        states_1 = states_0 + np.array([dx, dy, dz]) * dt
        return states_1

    def _four_wings_step(self, states_0: np.ndarray, dt: float) -> np.ndarray:
        dx = (self._attractor_params['a'] * states_0[0]) + (states_0[1] * states_0[2])
        dy = (
                (self._attractor_params['b'] * states_0[0]) +
                (self._attractor_params['c'] * states_0[1]) -
                (states_0[0] * states_0[2])
        )
        dz = -states_0[2] - (states_0[0] * states_0[1])

        states_1 = states_0 + np.array([dx, dy, dz]) * dt
        return states_1


class Attractors(ABC):
    def __init__(
            self,
            m_attractors: int,
            attractor_type: str,
            x0: Union[Sequence[np.ndarray], np.ndarray],
            attractor_params: Optional[Union[Sequence[dict], dict]] = None,
            noise_dist: dict = None,
    ):
        super().__init__()

        if isinstance(x0, np.ndarray):
            x0 = [x0.copy() for _ in range(m_attractors)]

        if isinstance(attractor_params, dict):
            attractor_params = [attractor_params.copy() for _ in range(m_attractors)]

        if attractor_params is None:
            attractor_params = [None for _ in range(m_attractors)]

        else:
            assert len(attractor_params) == m_attractors

        self.attractors = tuple(
            StrangeAttractor(
                attractor_type=attractor_type,
                attractor_params=attractor_params[i],
                x0=x0[i],
                noise_dist=noise_dist,
            )
            for i in range(m_attractors)
        )

        self._noise_dist = noise_dist
        self._m_attractors = m_attractors
        self.current_states = x0
        self.trajectories = None

    @staticmethod
    def _generate_noise(
            size: Sequence[int],
            dist: dict,
    ) -> np.ndarray:
        if dist['type'] == 'normal':
            noise = np.random.normal(loc=dist['loc'], scale=dist['scale'], size=size)

        elif dist['type'] == 'uniform':
            noise = np.random.uniform(low=dist['low'], high=dist['high'], size=size)

        else:
            noise = 0

        return noise

    def compute_trajectory(
            self,
            n: int,
            dt: float = 1.,
    ):
        self.trajectories = [
            self.next_state(dt=dt)
            for _ in range(n)
        ]

    def next_state(self, dt: float) -> List[np.ndarray]:
        next_states = [
            self.attractors[i].next_state(states_0=self.current_states[i], dt=dt)
            for i in range(self._m_attractors)
        ]

        if self._noise_dist is None:
            self.current_states = next_states

        else:
            self.current_states = next_states + self._generate_noise(
                size=(3,),
                dist=self._noise_dist,
            )

        return next_states

    @property
    def n_dynamics(self) -> int:
        return self._m_attractors

    @property
    def states_dim(self) -> int:
        return 3

    @property
    def observable_dim(self) -> int:
        return 3
