from abc import ABC
from torch.optim import Optimizer as TorchOptimizer
from typing import Dict, Sequence, Union, Any, Type, Iterator
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    StepLR,
    CyclicLR,
    ExponentialLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
)
from torch.nn.parameter import Parameter


class Optimizer(ABC):
    """
    A wrapper around PyTorch optimizers and schedulers.
    """

    def __init__(self, optimizers: Sequence[TorchOptimizer],
                 schedulers: Sequence[ReduceLROnPlateau] = None,
                 agnostic_schedulers: Sequence[
                     Union[StepLR, CyclicLR, ExponentialLR, CosineAnnealingLR, CosineAnnealingWarmRestarts]
                 ] = None):
        """
        Constructor for the wrapper _optimizer, which defines the optimizers to use,
        as well as any additional schedulers.

        :param optimizers: A sequence of optimizers on which the `.step` method will be
        called. Note that all optimizers should have already registered the parameters
        which they are responsible for optimizing.
        :param schedulers: A sequence of schedulers on which the `.step` method will be
        called, where each such call requires a scalar valued input, in order to decide
        whether to update the _optimizer parameters or not. Note that all schedulers
        should have already registered the optimizers whose parameters
        they are responsible for optimizing.
        :param agnostic_schedulers: A sequence of schedulers on which the `.step` method
        will be called, where each such call does NOT require a scalar valued input,
        in order to decide whether to update the _optimizer parameters or not.
        Note that all schedulers should have already registered the optimizers
        whose parameters they are responsible for optimizing.
        """

        self.optimizers = optimizers
        self.schedulers = schedulers
        self.agnostic_schedulers = agnostic_schedulers

    def zero_grad(self):
        """
        A method for zeroing all gradients of all optimizers

        :return: None
        """

        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def step(self):
        """
        A method for performing a step across all optimizers

        :return: None
        """

        for optimizer in self.optimizers:
            optimizer.step()

    def schedulers_step(self, val: float):
        """
        A method for performing a step across all schedulers

        :param val: a scalar value, used by the schedulers when performing their step.

        :return:None
        """

        if self.schedulers is not None:
            for s, sched in enumerate(self.schedulers):
                sched.step(val)

        if self.agnostic_schedulers is not None:
            for sched in self.agnostic_schedulers:
                sched.step()


class OptimizerInitializer(ABC):
    """
    An helper class, which takes as input the types and parameters of various
    optimizers and schedulers, and initialize all of them using a utility method.
     Useful for cross-validation experiments.
    """

    def __init__(
            self,
            optimizers_types: Sequence[Type[TorchOptimizer]],
            optimizers_params: Sequence[Dict[str, Any]],
            schedulers_types: Sequence[Sequence[Type[ReduceLROnPlateau]]] = None,
            schedulers_params: Sequence[Sequence[Dict[str, Any]]] = None,
            agnostic_schedulers_types: Sequence[
                Sequence[
                    Union[
                        StepLR, CyclicLR, ExponentialLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
                    ]
                ]
            ] = None,
            agnostic_schedulers_params: Sequence[Sequence[Dict[str, Any]]] = None,
    ):
        """
        :param optimizers_types: Sequence[Type[torch.optim.Optimizer]] - A sequence
        of the types of the optimizers to use.
        :param optimizers_params: Sequence[Dict[str, Any]] - A sequence of the
        parameters of the optimizers to use.
        :param schedulers_types: Sequence[Sequence[Type[ReduceLROnPlateau]]] - A
        sequence of sequences, where each element 'i' in the outer sequence,
        is a sequence containing the types of all schedulers
        to be applied to optimizer 'i'.
        :param schedulers_params: Sequence[Sequence[Dict[str, Any]]] - A sequence of
        sequences, where each element 'i' in the outer sequence, is a sequence
        containing the parameters of all schedulers to be applied to optimizer 'i'.
        :param agnostic_schedulers_types: Sequence[Sequence[Union[StepLR, CyclicLR,
        ExponentialLR]]] - A sequence of sequences, where each element 'i' in
        the outer sequence, is a sequence containing the types of all agnostic
        schedulers to be applied to optimizer 'i'.
        :param agnostic_schedulers_params: Sequence[Sequence[Dict[str, Any]]] - A
        sequence of sequences, where each element 'i' in the outer sequence,
        is a sequence containing the parameters of all agnostic schedulers to be
        applied to optimizer 'i'.
        """

        assert len(optimizers_params) == len(optimizers_types)

        if schedulers_types is not None:
            assert len(optimizers_params) == len(schedulers_types)
            assert len(schedulers_types) == len(schedulers_params)

        if agnostic_schedulers_types is not None:
            assert len(agnostic_schedulers_params) == len(agnostic_schedulers_types)
            assert len(agnostic_schedulers_types) == len(optimizers_types)

        self._optimizers_types = optimizers_types
        self._optimizers_params = optimizers_params
        self._schedulers_types = schedulers_types
        self._schedulers_params = schedulers_params
        self._agnostic_schedulers_types = agnostic_schedulers_types
        self._agnostic_schedulers_params = agnostic_schedulers_params

    def initialize(
            self,
            trainable_parameters: Sequence[Iterator[Parameter]],
    ) -> Optimizer:
        """
        An helper method which instantiate an 'Optimizer' class with all of the
        various optimizers and schedulers defined in the init, each time it is called.

        :param trainable_parameters: (Sequence[Generator[Parameter]]) A sequence of
        generators of trainable parameters, where each element 'i' in
        'trainable_parameters' should contained the trainable
        parameters for optimizer 'i' in self._optimizers_types.

        :return: (Optimizer) An 'Optimizer' wrapper class around all of the
        various PyTorch optimizers and schedulers defined in the init.
        """

        # Validate inputs
        assert len(trainable_parameters) == len(self._optimizers_params), \
            f"{len(trainable_parameters)} groups of trainable parameters where " \
            f"supplied to {len(self._optimizers_params)} optimizers."

        # Instantiate optimizers
        optimizers_params = [p.copy() for p in self._optimizers_params]
        for i in range(len(optimizers_params)):
            optimizers_params[i]['params'] = trainable_parameters[i]

        optimizers = [
            optim(**optimizers_params[i]) for i, optim in
            enumerate(self._optimizers_types)
        ]

        # Instantiate schedulers
        if self._schedulers_types is not None:
            schedulers_params = [[p.copy() for p in sp] for sp in self._schedulers_params]
            for i in range(len(schedulers_params)):
                for j in range(len(schedulers_params[i])):
                    schedulers_params[i][j]['optimizer'] = optimizers[i]

            schedulers_ = [
                [
                    scheduler(**schedulers_params[i][j]) for j, scheduler in enumerate(st)
                ]
                for i, st in enumerate(self._schedulers_types)
            ]
            schedulers = []
            for s in schedulers_:
                schedulers += s

        else:
            schedulers = None

        # Instantiate agnostic schedulers
        if self._agnostic_schedulers_types is not None:
            agnostic_schedulers_params = [
                [p.copy() for p in sp]
                for sp in self._agnostic_schedulers_params
            ]
            for i in range(len(agnostic_schedulers_params)):
                for j in range(len(agnostic_schedulers_params[i])):
                    agnostic_schedulers_params[i][j]['optimizer'] = optimizers[i]

            agnostic_schedulers_ = [
                [
                    # scheduler(**agnostic_schedulers_params[i][j])
                    scheduler(
                        agnostic_schedulers_params[i][j]['optimizer'],
                        *list(agnostic_schedulers_params[i][j].values())[:-1]
                    )
                    for j, scheduler in enumerate(st)
                ]
                for i, st in enumerate(self._agnostic_schedulers_types)
            ]

            agnostic_schedulers = []
            for s in agnostic_schedulers_:
                agnostic_schedulers += s

        else:
            agnostic_schedulers = None

        optimizer = Optimizer(
            optimizers=optimizers,
            schedulers=schedulers,
            agnostic_schedulers=agnostic_schedulers,
        )

        return optimizer
