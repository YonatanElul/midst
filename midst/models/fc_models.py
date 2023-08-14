from abc import ABC
from torch import Tensor
from typing import Union
from midst.models.activations import Snake, Swish
from midst.utils.defaults import MODELS_TENSOR_PREDICITONS_KEY, OTHER_KEY

import torch
import torch.nn as nn


def init_weights(module: nn.Module):
    """
    Utility method for initializing weights in layers

    :param module: (PyTorch Module) The layer to be initialized.

    :return: None
    """

    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)


def get_activation_layer(activation: Union[str, dict] = 'relu') -> nn.Module:
    """
    A utility method for defining and instantiating a PyTorch
    non-linear activation layer.

    :param activation: (str / dict / None) non-linear activation function to apply.
    If a string is given, using the layer with default parameters.
    if dict is given uses the 'name' key to determine which activation function to
    use and the 'params' key should have a dict with the required parameters as a
    key-value pairs. Currently supported activations: 'relu', 'gelu', 'elu',
    'hardshrink', 'leakyrelu', 'prelu', 'tanh', 'snake', default = 'relu'

    :return: (PyTorch Module) The activation layer.
    """

    # Define the activations keys-modules pairs
    activations_dict = {
        'relu': nn.ReLU,
        'gelu': nn.GELU,
        'elu': nn.ELU,
        'hardshrink': nn.Hardshrink,
        'leakyrelu': nn.LeakyReLU,
        'prelu': nn.PReLU,
        'tanh': nn.Tanh,
        'snake': Snake,
        'swish': Swish,
    }

    # Validate inputs
    if (activation is not None and not isinstance(activation, str)
            and not isinstance(activation, dict)):
        raise ValueError("Can't take specs for the activation layer of type"
                         f" {type(activation)}, please specify either with a "
                         "string or a dictionary.")

    if (isinstance(activation, dict)
            and activation['name'] not in activations_dict.keys()):
        raise ValueError(f"{activation['name']} is not a supported Activation type.")

    if isinstance(activation, str):
        activation_block = activations_dict[activation]()

    else:
        activation_block = activations_dict[activation['name']](
            **activation['params'])

    return activation_block


def get_normalization_layer(norm: dict) -> nn.Module:
    """
    A utility method for defining and instantiating a PyTorch normalization layer.

    :param norm: (dict / None) Denotes the normalization layer to use with the FC layer.
    The dict should contain at least two keys, 'name' for indicating the type of
    normalization to use, and 'params', which should also map to a dict with all
    required parameters for the normalization layer. At the minimum, the 'params' dict
    should define the 'num_channels' key to indicate the expected number of
    channels on which to apply the normalization. For the GroupNorm, it is also
    required to specify a 'num_groups' key.
    If None then doesn't add normalization layer.
    Currently supported normalization layers: 'batch1d', 'batch2d', 'batch3d',
    'instance1d', 'instance2d', 'instance3d', 'group', where 'batch' stands for
    BatchNorm, `instance` stands for InstanceNorm and `group` stands
    for GroupNorm. Default == None.

    :return: (PyTorch Module) The normalization layer.
    """

    # Define the normalizations keys-modules pairs
    norms_dict = {
        'batch1d': nn.BatchNorm1d,
        'batch2d': nn.BatchNorm2d,
        'batch3d': nn.BatchNorm3d,
        'instance1d': nn.InstanceNorm1d,
        'instance2d': nn.InstanceNorm2d,
        'instance3d': nn.InstanceNorm3d,
        'layer': nn.LayerNorm,
        'group': nn.GroupNorm,
    }

    # Validate inputs
    if norm is not None and not isinstance(norm, dict):
        raise ValueError(f"Can't specify norm as a {type(norm)} type. Please "
                         f"either use a dict, or None.")

    if (isinstance(norm, dict)
            and norm['name'] not in norms_dict.keys()):
        raise ValueError(f"{norm['name']} is not a supported Normalization type.")

    norm_block = norms_dict[norm['name']](
        **norm['params'])

    return norm_block


def get_fc_layer(
        input_dim: int,
        output_dim: int,
        bias: bool = False,
        activation: Union[str, dict, None] = 'relu',
        dropout: Union[float, None] = None,
        norm: Union[dict, None] = None,
        mc_dropout: bool = False,
) -> nn.Module:
    """
    A utility method for generating a FC layer

    :param input_dim: (int) input dimension of the 2D matrices
    (M for N X M matrix)
    :param output_dim: (int) output dimension of the 2D matrices
     (M for N X M matrix)
    :param bias: (bool) whether to use a bias in the FC layer or not,
     default = False
    :param activation: (str / dict / None) non-linear activation function to apply.
    If a string is given, using the layer with default parameters.
    if dict is given uses the 'name' key to determine which activation function to
    use and the 'params' key should have a dict with the required parameters as a
    key-value pairs. Currently supported activations: 'relu', 'gelu', 'elu',
    'hardshrink', 'leakyrelu', 'prelu', 'tanh', default = 'relu'
    :param dropout: (float/ None) rate of dropout to apply to the FC layer,
    if None than doesn't apply dropout, default = None
    :param norm: (dict / None) Denotes the normalization layer to use with the FC layer.
    The dict should contain at least two keys, 'name' for indicating the type of
    normalization to use, and 'params', which should also map to a dict with all
    required parameters for the normalization layer. At the minimum, the 'params' dict
    should define the 'num_channels' key to indicate the expected number of
    channels on which to apply the normalization. For the GroupNorm, it is also
    required to specify a 'num_groups' key.
    If None then doesn't add normalization layer.
    Currently supported normalization layers: 'batch1d', 'batch2d', 'batch3d',
    'instance1d', 'instance2d', 'instance3d', 'group', where 'batch' stands for
    BatchNorm, `instance` stands for InstanceNorm and `group` stands
    for GroupNorm. Default == None.
    :param mc_dropout: (bool) Whether to use MC dropout or not

    :return: (PyTorch Module) the instantiated layer, according to the given specs
    """

    # Validate inputs
    if dropout is not None and (not isinstance(dropout, float) and not isinstance(dropout, int)):
        raise ValueError(f"Can't specify dropout as a {type(dropout)} type. Please "
                         f"either use float, or None.")

    # Add the FC block
    blocks = []
    fc_block = nn.Linear(
        in_features=input_dim,
        out_features=output_dim,
        bias=bias,
    )
    blocks.append(fc_block)

    # Add the Normalization block if required
    if norm is not None:
        norm_block = get_normalization_layer(norm)
        blocks.append(norm_block)

    # Add the Activation block if required
    if activation is not None:
        activation_block = get_activation_layer(activation)
        blocks.append(activation_block)

    # Add the Dropout block if required
    if dropout is not None:
        if mc_dropout:
            dropout_block = MCDropout(p=dropout)

        else:
            dropout_block = nn.Dropout(p=dropout)

        blocks.append(dropout_block)

    # Encapsulate all blocks as a single `Sequential` module.
    fc_layer = nn.Sequential(*blocks)

    return fc_layer


class SkipConnectionFCBlock(nn.Module):
    """
    A FC block which applies skip connection.
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            bias: bool = False,
            activation: Union[str, dict, None] = 'relu',
            dropout: Union[float, None] = None,
            norm: Union[dict, None] = None,
            mc_dropout: bool = False,
    ):

        super(SkipConnectionFCBlock, self).__init__()

        self._fc_block = get_fc_layer(
            input_dim=input_dim,
            output_dim=output_dim,
            bias=bias,
            activation=activation,
            dropout=dropout,
            norm=norm,
            mc_dropout=mc_dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self._fc_block(x)
        out = out + x
        return out

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


class SpectralNorm(nn.Module):
    def __init__(
            self,
            weight: Tensor,
            n_power_iterations: int = 1,
            dim: int = 0,
            eps: float = 1e-12
    ) -> None:
        super().__init__()
        ndim = weight.ndim
        if dim >= ndim or dim < -ndim:
            raise IndexError("Dimension out of range (expected to be in range of "
                             f"[-{ndim}, {ndim - 1}] but got {dim})")

        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))

        self.dim = dim if dim >= 0 else dim + ndim
        self.eps = eps
        if ndim > 1:
            # For ndim == 1 we do not need to approximate anything (see _SpectralNorm.forward)
            self.n_power_iterations = n_power_iterations
            weight_mat = self._reshape_weight_to_matrix(weight)
            h, w = weight_mat.size()

            u = weight_mat.new_empty(h).normal_(0, 1)
            v = weight_mat.new_empty(w).normal_(0, 1)
            self.register_buffer('_u', nn.functional.normalize(u, dim=0, eps=self.eps))
            self.register_buffer('_v', nn.functional.normalize(v, dim=0, eps=self.eps))

            # Start with u, v initialized to some reasonable values by performing a number
            # of iterations of the power method
            self._power_method(weight_mat, 15)

    def _reshape_weight_to_matrix(self, weight: Tensor) -> Tensor:
        # Precondition
        assert weight.ndim > 1

        if self.dim != 0:
            # permute dim to front
            weight = weight.permute(self.dim, *(d for d in range(weight.dim()) if d != self.dim))

        return weight.flatten(1)

    @torch.autograd.no_grad()
    def _power_method(self, weight_mat: torch.Tensor, n_power_iterations: int) -> None:
        # See original note at torch/nn/utils/spectral_norm.py
        # NB: If `do_power_iteration` is set, the `u` and `v` vectors are
        #     updated in power iteration **in-place**. This is very important
        #     because in `DataParallel` forward, the vectors (being buffers) are
        #     broadcast from the parallelized module to each module replica,
        #     which is a new module object created on the fly. And each replica
        #     runs its own spectral norm power iteration. So simply assigning
        #     the updated vectors to the module this function runs on will cause
        #     the update to be lost forever. And the next time the parallelized
        #     module is replicated, the same randomly initialized vectors are
        #     broadcast and used!
        #
        #     Therefore, to make the change propagate back, we rely on two
        #     important behaviors (also enforced via tests):
        #       1. `DataParallel` doesn't clone storage if the broadcast tensor
        #          is already on correct device; and it makes sure that the
        #          parallelized module is already on `device[0]`.
        #       2. If the out tensor in `out=` kwarg has correct shape, it will
        #          just fill in the values.
        #     Therefore, since the same power iteration is performed on all
        #     devices, simply updating the tensors in-place will make sure that
        #     the module replica on `device[0]` will update the _u vector on the
        #     parallized module (by shared storage).
        #
        #    However, after we update `u` and `v` in-place, we need to **clone**
        #    them before using them to normalize the weight. This is to support
        #    backproping through two forward passes, e.g., the common pattern in
        #    GAN training: loss = D(real) - D(fake). Otherwise, engine will
        #    complain that variables needed to do backward for the first forward
        #    (i.e., the `u` and `v` vectors) are changed in the second forward.

        # Precondition
        assert weight_mat.ndim > 1
        self._v = self._v.to(weight_mat.device)
        self._u = self._u.to(weight_mat.device)

        for _ in range(n_power_iterations):
            # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
            # are the first left and right singular vectors.
            # This power iteration produces approximations of `u` and `v`.
            self._u = nn.functional.normalize(
                torch.mv(weight_mat, self._v),
                dim=0,
                eps=self.eps,
                out=self._u,
            )
            self._v = nn.functional.normalize(
                torch.mv(weight_mat.t(), self._u),
                dim=0,
                eps=self.eps,
                out=self._v,
            )

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.ndim == 1:
            # Faster and more exact path, no need to approximate anything
            return nn.functional.normalize(weight, dim=0, eps=self.eps)

        else:
            weight_mat = self._reshape_weight_to_matrix(weight)
            if self.training:
                self._power_method(weight_mat, self.n_power_iterations)
            # See above on why we need to clone
            u = self._u.clone(memory_format=torch.contiguous_format)
            v = self._v.clone(memory_format=torch.contiguous_format)
            # The proper way of computing this should be through F.bilinear, but
            # it seems to have some efficiency issues:
            # https://github.com/pytorch/pytorch/issues/58093
            sigma = torch.dot(u, torch.mv(weight_mat, v))
            return weight / sigma

    def right_inverse(self, value: torch.Tensor) -> torch.Tensor:
        # we may want to assert here that the passed value already
        # satisfies constraints
        return value


class FCEncoderDecoder(nn.Module, ABC):
    """
    A fully connected encoder/decoder model.
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int = 64,
            n_layers: int = 8,
            l0_units: int = 1024,
            units_factor: float = 0.5,
            units_per_layer: tuple = None,
            activation: Union[str, dict] = 'relu',
            final_activation: Union[str, dict] = None,
            norm: Union[str, dict] = None,
            dropout: float = None,
            bias: bool = False,
            spectral_norm: bool = False,
            mc_dropout: bool = False,
            skip_connections: bool = False,
    ):
        """
        The constructor of the FCEncoderDecoder class, note that this class can
        save as both encoder and decoder FC-based models.

        :param input_dim: (int) Either the dimensionality of the input 2D matrix,
        i.e. if the inputs is a matrix of size m X n, then `input_dim` is n, when
        using the model as an encoder, or the dimensionality of the latent
        representation in the embedding space when using the model as a decoder.
        :param output_dim: (int) Required dimensionality for the latent embeddings when
        using the model as an encoder, or the dimensionality of the final output
        when using the model as a decoder.
        :param n_layers: (int) Number of FC layers to include in the encoder.
        :param l0_units: (int) Number of units to include in the first FC layer.
        :param units_factor: (float) Multiplicative factor for reducing/increasing
        the number of units in each consecutive FC layer when using the model as
        encoder / decoder.
        :param units_per_layer: (tuple) A tuple with 'n_layers' elements, where each
        element i is an integer indicating  the number of units to use in the i-th FC
        layer. Must specify either 'units_per_layer',
        or 'l0_units' and 'units_down_factor'.
        :param bias: (bool) whether to use a bias in the FC layer or not,
         default = False
        :param activation: (str / dict / None) non-linear activation function to apply.
        If a string is given, using the layer with default parameters.
        if dict is given uses the 'name' key to determine which activation function to
        use and the 'params' key should have a dict with the required parameters as a
        key-value pairs. Currently supported activations: 'relu', 'gelu', 'elu',
        'hardshrink', 'leakyrelu', 'prelu', 'tanh', default = 'relu'.
        :param final_activation: (str / dict / None) non-linear activation function
        to apply to the final layer.
        :param dropout: (float/ None) rate of dropout to apply to the FC layer,
        if None than doesn't apply dropout, default = None
        :param norm: (dict / None) Denotes the normalization layer to use with the
        FC layer. The dict should contains at least two keys, 'name' for indicating
        the type of normalization to use, and 'params', which should also map to a dict
        with all required parameters for the normalization layer. At the minimum,
        the 'params' dict should define the 'num_channels' key to indicate the expected
        number of channels on which to apply the normalization. For the GroupNorm,
        it is also required to specify a 'num_groups' key.
        If None then doesn't add normalization layer.
        Currently supported normalization layers: 'batch1d', 'batch2d', 'batch3d',
        'instance1d', 'instance2d', 'instance3d', 'group', where 'batch' stands for
        BatchNorm, `instance` stands for InstanceNorm and `group` stands
        for GroupNorm. Default == None.
        :param mc_dropout: (bool) Whether to use MC dropout or not
        :param skip_connections: (bool) Whether to use skip connections,
        making the number of units to be l0_units in all layers
        """

        super(FCEncoderDecoder, self).__init__()

        self._input_dim = input_dim
        self._output_dim = output_dim
        self._spectral_norm = spectral_norm
        spectral_norms = []

        # Validate inputs
        assert ((units_per_layer is not None and units_factor is None) or
                units_per_layer is None and units_factor is not None and
                l0_units is not None), \
            f"Cannot specify both 'units_per_layer' = {units_per_layer} and" \
            f" 'units_factor' = {units_factor}, 'l0_units' = {l0_units}. " \
            "Please specify either 'units_per_layer' or " \
            "'units_factor' and 'l0_units'."

        if units_per_layer is not None:
            assert len(units_per_layer) == n_layers, \
                f"If 'units_per_layer' is not None, then it should specify the " \
                f"# units for every layer," \
                f" however {len(units_per_layer)} specification are" \
                f" given for {n_layers} layers."

            out_dim = units_per_layer[0]

        else:
            out_dim = l0_units

        if isinstance(norm, str) and norm in (
                'batch1d', 'batch2d', 'batch3d',
                'instance1d', 'instance2d', 'instance3d'
        ):
            norm = {
                'name': norm,
                'params': input_dim,
            }

        elif isinstance(norm, dict) and ('name' not in norm.keys() or
                                         'params' not in norm.keys()):
            raise ValueError(
                "If norm is a dict, it must contain the 'name' and 'params' keys."
            )

        elif norm is not None and not isinstance(norm, dict):
            raise ValueError(
                "norm must be either a string of: 'batch1d', 'batch2d', 'batch3d',"
                " 'instance1d', 'instance2d', 'instance3d', or None, or a dict"
            )

        # Build model
        layers = []

        if skip_connections:
            layers.append(
                get_fc_layer(
                    input_dim=input_dim,
                    output_dim=out_dim,
                    bias=bias,
                    activation=activation,
                    dropout=dropout,
                    mc_dropout=mc_dropout,
                )
            )

        n_inner_layers = n_layers - 2 if skip_connections else n_layers - 1
        in_dim = out_dim if skip_connections else input_dim
        for i in range(n_inner_layers):
            if norm is not None and norm['name'] == 'layer':
                if isinstance(norm['params']['normalized_shape'], int):
                    norm['params']['normalized_shape'] = out_dim

                else:
                    norm['params']['normalized_shape'][-1] = out_dim

            if skip_connections:
                layers.append(
                    SkipConnectionFCBlock(
                        input_dim=in_dim,
                        output_dim=out_dim,
                        bias=bias,
                        activation=activation,
                        dropout=dropout,
                        norm=norm,
                        mc_dropout=mc_dropout,
                    )
                )

            else:
                layers.append(
                    get_fc_layer(
                        input_dim=in_dim,
                        output_dim=out_dim,
                        bias=bias,
                        activation=activation,
                        dropout=dropout,
                        norm=norm,
                        mc_dropout=mc_dropout,
                    )
                )
                spectral_norms.append(
                    SpectralNorm(
                        weight=layers[-1]._modules['0'].weight,
                        n_power_iterations=1,
                    )
                )

            in_dim = out_dim
            if norm is not None and norm['name'] != 'layer':
                norm = {
                    'params': in_dim,
                }

            if units_per_layer is not None and not skip_connections:
                out_dim = units_per_layer[i + 1]

            elif i > 0 and i % 2 == 0 and not skip_connections:
                out_dim = int(in_dim * units_factor)

        if norm is not None and norm['name'] == 'layer':
            if isinstance(norm['params']['normalized_shape'], int):
                norm['params']['normalized_shape'] = output_dim

            else:
                norm['params']['normalized_shape'][-1] = output_dim

        layers.append(
            get_fc_layer(
                input_dim=in_dim,
                output_dim=output_dim,
                bias=bias,
                activation=final_activation,
                mc_dropout=mc_dropout,
            )
        )

        self._model = nn.Sequential(*layers)

        if spectral_norm:
            self._sn = spectral_norms

        # Initialize weights
        self._model.apply(init_weights)

    def forward(self, x: Tensor) -> Tensor:
        """
        The forward logic for the 'FCEncoderDecoder' class.

        :param x: (Tensor) The input tensor.

        :return: (Tensor) The resulting tensor from the forward pass.
        """

        if self.training and self._spectral_norm:
            with torch.no_grad():
                for i, s in enumerate(self._sn):
                    self._model._modules[f'{i}']._modules['0'].weight = nn.Parameter(
                        s(self._model._modules[f'{i}']._modules['0'].weight),
                        requires_grad=True,
                    )

        outputs = self._model(x)

        return outputs

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


class EncoderDecoderPredictor(FCEncoderDecoder):
    """
    A wrapper around FCEncoderDecoder which allows for controlling the prediction horizon
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            prediction_horizon: int,
            n_layers: int = 4,
            l0_units: int = 1024,
            units_factor: float = 0.5,
            units_per_layer: tuple = None,
            activation: Union[str, dict] = 'relu',
            final_activation: Union[str, dict] = None,
            norm: Union[str, dict] = None,
            dropout: float = None,
            bias: bool = False,
    ):
        super(EncoderDecoderPredictor, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            l0_units=l0_units,
            units_factor=units_factor,
            units_per_layer=units_per_layer,
            activation=activation,
            final_activation=final_activation,
            norm=norm,
            dropout=dropout,
            bias=bias,
        )

        self._prediction_horizon = prediction_horizon

    def forward(self, x: Tensor) -> Tensor:
        """
        The forward logic for the 'EncoderDecoderPredictor' class.

        :param x: (Tensor) The input tensor.

        :return: (Tensor) The resulting tensor from the forward pass.
        """

        # Output have a shape of (N, M, T, self._output_dim), where:
        # N = batch size, M = # systems, T = temporal trajectory length
        outputs = super().forward(x)

        # Re-shape it to (N, M, T * H, D), where:
        # N = batch size, M = # systems, T = temporal trajectory length,
        # H = prediction horizon, D = state dimensionality, and the H predictions from each time point t, are
        # ordered consecutively at axis 2.
        outputs = outputs.reshape(
            outputs.shape[0],
            outputs.shape[1],
            (outputs.shape[2] * self._prediction_horizon),
            (outputs.shape[3] // self._prediction_horizon)
        )

        output = {
            MODELS_TENSOR_PREDICITONS_KEY: outputs,
            OTHER_KEY: {},
        }

        return output


class FCAE(nn.Module, ABC):
    """
    A fully connected auto-encoder model.
    Where both the encoder and the decoder are based on the FCEncoderDecoder class.
    """

    def __init__(
            self,
            input_dim: int,
            embedding_dim: int,
            output_dim: int = 64,
            n_layers_encoder: int = 8,
            n_layers_decoder: int = 8,
            l0_units: int = 1024,
            units_factor: float = 0.5,
            activation: Union[str, dict] = 'relu',
            final_activation: Union[str, dict] = None,
            norm: str = None,
            dropout: float = None,
            bias: bool = False,
    ):
        """
        The constructor of the FCAE class, which is a composition of to symmetric FCEncoderDecoder models, one
        serving as an encoder and one as a decoder.

        :param input_dim: (int) Dimensionality of the inputs.
        :param embedding_dim: (int) Dimensionality of the latent embedding.
        :param output_dim: (int) Dimensionality of the outputs.
        :param n_layers_encoder: (int) Number of FC layers to include in the encoder.
        :param n_layers_encoder: (int) Number of FC layers to include in the decoder.
        """

        super(FCAE, self).__init__()

        self._encoder = FCEncoderDecoder(
            input_dim=input_dim,
            output_dim=embedding_dim,
            n_layers=n_layers_encoder,
            l0_units=l0_units,
            units_factor=units_factor,
            activation=activation,
            final_activation=final_activation,
            norm=norm,
            dropout=dropout,
            bias=bias,
        )

        self._decoder = FCEncoderDecoder(
            input_dim=embedding_dim,
            output_dim=output_dim,
            n_layers=n_layers_decoder,
            l0_units=int(l0_units * (units_factor ** (n_layers_encoder - 1))),
            units_factor=int(1 / units_factor),
            activation=activation,
            final_activation=final_activation,
            norm=norm,
            dropout=dropout,
            bias=bias,
        )

        # Initialize weights
        self._encoder.apply(init_weights)
        self._decoder.apply(init_weights)

    def forward(self, x: Tensor) -> dict:
        """
        The forward logic for the 'FCAE' class.

        :param x: (Tensor) The input tensor.

        :return: (Tensor) The resulting tensor from the forward pass.
        """

        embeddings = self._encoder(x)
        predictions = self._decoder(embeddings)

        output = {
            MODELS_TENSOR_PREDICITONS_KEY: predictions,
            OTHER_KEY: {
            },
        }

        return output

    def __call__(self, x: Tensor) -> dict:
        return self.forward(x)


class MLP(nn.Module, ABC):
    """
    A general MLP class
    """

    def __init__(self,
                 n_layers: int,
                 in_channels: int,
                 out_channels: int,
                 l0_units: int = 1024,
                 units_grow_rate: int = 2,
                 grow_every_x_layer: int = 2,
                 bias: bool = False,
                 activation: Union[str, dict, None] = 'relu',
                 final_activation: Union[str, dict, None] = None,
                 norm: Union[dict, None] = None,
                 final_norm: Union[dict, None] = None,
                 dropout: Union[float, None] = None):
        """
        The constructor for the MLP class.

        :param n_layers: (int) Number of FC layers to include in the encoder.
        :param in_channels: (int) input dimension of the 2D matrices
        (M for a N X M matrix) to the MLP.
        :param out_channels: (int) output dimension of the 2D matrices
        (M for a N X M matrix) from the MLP.
        :param l0_units: (int) Number of units to include in the first FC layer.
        :param units_grow_rate: (int) The factor by which to increase the number of
        units between FC layers.
        :param grow_every_x_layer: (int) Indicate after how many layers to increase the
        number of units by a factor of 'grow_every_x_layer'.
        :param bias: (bool) whether to use a bias in the MLP layers or not,
        default = False.
        :param activation: (str / dict / None) The non-linear activation function
        to apply across all layers of the MLP, except for last one.
        If a string is given, using the layer with default parameters.
        if dict is given uses the 'name' key to determine which activation function to
        use and the 'params' key should have a dict with the required parameters as a
        key-value pairs. Currently supported activations: 'relu', 'gelu', 'elu',
        'hardshrink', 'leakyrelu', 'prelu', 'tanh', default = 'relu'.
        :param final_activation: (str / dict / None) The non-linear activation function
        to apply only to the last layer. Takes in the same values as 'activation'.
        :param dropout: (float/ None) rate of dropout to apply across all MLP layers,
        if None than doesn't apply dropout, default = None.
        :param norm: (dict / None) Denotes the normalization layer to apply to all
         MLP layers, except for the last one.
        The dict should contains at least two keys, 'name' for indicating the type of
        normalization to use, and 'params', which should also map to a dict with all
        required parameters for the normalization layer. At the minimum, the 'params'
        dict should define the 'num_channels' key to indicate the expected number of
        channels on which to apply the normalization. For the GroupNorm, it is also
        required to specify a 'num_groups' key.
        If None then doesn't add normalization layer.
        Currently supported normalization layers: 'batch1d', 'batch2d', 'batch3d',
        'instance1d', 'instance2d', 'instance3d', 'group', where 'batch' stands for
        BatchNorm, `instance` stands for InstanceNorm and `group` stands
        for GroupNorm. Default == None.
        :param final_norm: (dict / None) Denotes the normalization layer to apply only
        to the last MLP layer. The dict should contains at least two keys, 'name' for
        indicating the type of normalization to use, and 'params', which should also
        map to a dict with all required parameters for the normalization layer.
        At the minimum, the 'params' dict should define the 'num_channels' key to
        indicate the expected number of channels on which to apply the normalization.
        For the GroupNorm, it is also required to specify a 'num_groups' key.
        If None then doesn't add normalization layer.
        Currently supported normalization layers: 'batch1d', 'batch2d', 'batch3d',
        'instance1d', 'instance2d', 'instance3d', 'group', where 'batch' stands for
        BatchNorm, `instance` stands for InstanceNorm and `group` stands
        for GroupNorm. Default == None.
        """

        super(MLP, self).__init__()

        # Define the first layer
        fc_layers = [
            get_fc_layer(
                input_dim=in_channels,
                output_dim=l0_units,
                bias=bias,
                activation=activation,
                dropout=dropout,
                norm=norm,
            ),
        ]

        # Define all layers except the first and last
        n_units = l0_units
        in_units = out_units = n_units
        for layer in range(1, (n_layers - 1)):

            # Increase the number of units every x layer
            if layer % grow_every_x_layer == 0:
                out_units *= units_grow_rate

            fc_layers.append(
                get_fc_layer(
                    input_dim=in_units,
                    output_dim=out_units,
                    bias=bias,
                    activation=activation,
                    dropout=dropout,
                    norm=norm,
                )
            )

            in_units = out_units

        # Define the last layer
        fc_layers.append(
            get_fc_layer(
                input_dim=in_units,
                output_dim=out_channels,
                bias=bias,
                activation=final_activation,
                dropout=None,
                norm=final_norm,
            )
        )

        self._model = nn.Sequential(*fc_layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        The forward logic for the MLP model.

        :param x: (Tensor) contains the raw inputs.

        :return: (Tensor) the output of the forward pass of the MLP.
        """

        out = self._model(x)

        return out

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


class MCDropout(nn.Module):
    """
    A wrapper around PyTorch Dropout layer, which always uses the layer in 'train mode'
    """

    def __init__(self, p: float = 0.):
        """
        :param p: (float) rate of dropout. Defaults to 0.
        """

        super().__init__()

        self._dropout = nn.Dropout(p=p)
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        self.train(True)
        return self._dropout(x)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


