import torch
import torch.ao.nn.quantized as nnq
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, input):
        # gru_input shape: (batch, seq_len, hidden_size)
        attention_weights = F.softmax(self.attention(input), dim=1)
        context_vector = torch.sum(attention_weights * input, dim=1)
        return context_vector, attention_weights
    

class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1)

    def forward(self, x):
        return self._conv_forward(F.pad(x, [self.causal_padding, 0]), self.weight, self.bias)

class CausalConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1) + self.output_padding[0] + 1 - self.stride[0]
    
    def forward(self, x, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose1d')

        assert isinstance(self.padding, tuple)
        output_padding = self._output_padding(
            x, output_size, self.stride, self.padding, self.kernel_size, self.dilation)
        return F.conv_transpose1d(
            x, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)[...,:-self.causal_padding]    

def _make_divisible(v, divisor=8, min_value=None):
    """
    This function is taken from the original tf repo. It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py

    It ensures that all layers have a channel number that is divisible by divisor.

    Arguments
    ---------
    v : int
        The original number of channels.
    divisor : int, optional
        The divisor to ensure divisibility (default is 8).
    min_value : int or None, optional
        The minimum value for the divisible channels (default is None).

    Returns
    -------
    int
        The adjusted number of channels.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def correct_pad(input_shape, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.

    Arguments
    ---------
    input_shape : tuple or list
        Shape of the input tensor (height, width).
    kernel_size : int or tuple
        Size of the convolution kernel.

    Returns
    -------
    tuple
        A tuple representing the zero-padding in the format (left, right, top, bottom).
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_shape[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_shape[0] % 2, 1 - input_shape[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return (
        int(correct[1] - adjust[1]),
        int(correct[1]),
        int(correct[0] - adjust[0]),
        int(correct[0]),
    )


def preprocess_input(x, **kwargs):
    """Normalize input channels between [-1, 1].

    Arguments
    ---------
    x : torch.Tensor
        Input tensor to be preprocessed.

    Returns
    -------
    torch.Tensor
        Normalized tensor with values between [-1, 1].
    """

    return (x / 128.0) - 1


def get_xpansion_factor(t_zero, beta, block_id, num_blocks):
    """Compute the expansion factor based on the formula from the paper.

    Arguments
    ---------
    t_zero : float
        The base expansion factor.
    beta : float
        The shape factor.
    block_id : int
        The identifier of the current block.
    num_blocks : int
        The total number of blocks.

    Returns
    -------
    float
        The computed expansion factor.
    """
    return (t_zero * beta) * block_id / num_blocks + t_zero * (
        num_blocks - block_id
    ) / num_blocks


class ReLUMax(torch.nn.Module):
    """Implements ReLUMax.

    Arguments
    ---------
    max_value : float
        The maximum value for the clamp operation.

    """

    def __init__(self, max):
        super(ReLUMax, self).__init__()
        self.max = max

    def forward(self, x):
        """Forward pass of ReLUMax.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after applying ReLU with max value.
        """
        return torch.clamp(x, min=0, max=self.max)


class SEBlock(torch.nn.Module):
    """Implements squeeze-and-excitation block.

    Arguments
    ---------
    in_channels : int
        Input number of channels.
    out_channels : int
        Output number of channels.
    h_swish : bool, optional
        Whether to use the h_swish (default is True).

    """

    def __init__(self, in_channels, out_channels, h_swish=True):
        super(SEBlock, self).__init__()

        self.se_conv = CausalConv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            bias=False,
        )

        self.se_conv2 = CausalConv1d(
            out_channels, in_channels, kernel_size=1, bias=False, padding=0
        )

        if h_swish:
            self.activation = nn.Hardswish(inplace=True)
        else:
            self.activation = ReLUMax(6)

        # It serves for the quantization.
        # The behavior remains equivalent for the unquantized models.
        self.mult = nnq.FloatFunctional()

    def forward(self, x):
        """Executes the squeeze-and-excitation block.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output of the squeeze-and-excitation block.
        """

        inp = x
        x = F.adaptive_avg_pool1d(x, 1)
        x = self.se_conv(x)
        x = self.activation(x)
        x = self.se_conv2(x)
        x = torch.sigmoid(x)

        return self.mult.mul(inp, x)  # Equivalent to ``torch.mul(a, b)``


class DepthwiseCausalConv(CausalConv1d):
    """Depthwise Causal 1D convolution layer.

    Arguments
    ---------
    in_channels : int
        Number of input channels.
    depth_multiplier : int, optional
        The channel multiplier for the output channels (default is 1).
    kernel_size : int or tuple, optional
        Size of the convolution kernel (default is 3).
    stride : int or tuple, optional
        Stride of the convolution (default is 1).
    padding : int or tuple, optional
        Zero-padding added to both sides of the input (default is 0).
    dilation : int or tuple, optional
        Spacing between kernel elements (default is 1).
    bias : bool, optional
        If True, adds a learnable bias to the output (default is False).
    padding_mode : str, optional
        'zeros' or 'circular'. Padding mode for convolution (default is 'zeros').

    """

    def __init__(
        self,
        in_channels,
        depth_multiplier=1,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        padding_mode="zeros",
    ):
        out_channels = in_channels * depth_multiplier
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode,
        )


class SeparableCausalConv1d(torch.nn.Module):
    """Implements SeparableCausalConv1d.

    Arguments
    ---------
    in_channels : int
        Input number of channels.
    out_channels : int
        Output number of channels.
    activation : function, optional
        Activation function to apply (default is torch.nn.functional.relu).
    kernel_size : int, optional
        Kernel size (default is 3).
    stride : int, optional
        Stride for convolution (default is 1).
    padding : int, optional
        Padding for convolution (default is 0).
    dilation : int, optional
        Dilation factor for convolution (default is 1).
    bias : bool, optional
        If True, adds a learnable bias to the output (default is True).
    padding_mode : str, optional
        Padding mode for convolution (default is 'zeros').
    depth_multiplier : int, optional
        Depth multiplier (default is 1).

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        activation=torch.nn.functional.relu,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        padding_mode="zeros",
        depth_multiplier=1,
    ):
        super().__init__()

        self._layers = torch.nn.ModuleList()

        depthwise = CausalConv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=1,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

        spatialConv = CausalConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=dilation,
            # groups=in_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

        bn = torch.nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.999)

        self._layers.append(depthwise)
        self._layers.append(spatialConv)
        self._layers.append(bn)
        self._layers.append(activation)

    def forward(self, x):
        """Executes the SeparableConv2d block.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output of the convolution.
        """
        for layer in self._layers:
            x = layer(x)

        return x


class PhiNetCausalConvBlock(nn.Module):
    """Implements PhiNet's convolutional block.

    Arguments
    ---------
    in_shape : tuple
        Input shape of the conv block.
    expansion : float
        Expansion coefficient for this convolutional block.
    stride: int
        Stride for the conv block.
    filters : int
        Output channels of the convolutional block.
    block_id : int
        ID of the convolutional block.
    has_se : bool
        Whether to include use Squeeze and Excite or not.
    res : bool
        Whether to use the residual connection or not.
    h_swish : bool
        Whether to use HSwish or not.
    k_size : int
        Kernel size for the depthwise convolution.

    """

    def __init__(
        self,
        in_channels,
        expansion,
        stride,
        filters,
        has_se,
        block_id=None,
        res=True,
        h_swish=True,
        k_size=3,
        dp_rate=0.05,
        divisor=1,
        dilation=1,
    ):
        super(PhiNetCausalConvBlock, self).__init__()

        self.param_count = 0

        self.skip_conn = False
        
        self._layers = torch.nn.ModuleList()
        # in_channels = in_shape[0]
        
        # Define activation function
        if h_swish:
            activation = nn.Hardswish(inplace=True)
        else:
            activation = ReLUMax(6)

        if block_id:
            # Expand
            conv1 = CausalConv1d(
                in_channels,
                _make_divisible(int(expansion * in_channels), divisor=divisor),
                kernel_size=1,
                padding=0,
                bias=False,
            )

            bn1 = nn.BatchNorm1d(
                _make_divisible(int(expansion * in_channels), divisor=divisor),
                eps=1e-3,
                momentum=0.999,
            )

            self._layers.append(conv1)
            self._layers.append(bn1)
            self._layers.append(activation)

        if stride == 2:
            padding = correct_pad([res, res], 3)

        self._layers.append(nn.Dropout1d(dp_rate))

        d_mul = 1
        in_channels_dw = (
            _make_divisible(int(expansion * in_channels), divisor=divisor)
            if block_id
            else in_channels
        )
        out_channels_dw = in_channels_dw * d_mul
        dw1 = DepthwiseCausalConv(
            in_channels=in_channels_dw,
            depth_multiplier=d_mul,
            kernel_size=k_size,
            stride=stride,
            bias=False,
            padding= 0,#k_size // 2 if stride == 1 else (padding[1], padding[3]),
            dilation = dilation,
        )

        bn_dw1 = nn.BatchNorm1d(
            out_channels_dw,
            eps=1e-3,
            momentum=0.999,
        )

        # It is necessary to reinitialize the activation
        # for functions using Module.children() to work properly.
        # Module.children() does not return repeated layers.
        if h_swish:
            activation = nn.Hardswish(inplace=True)
        else:
            activation = ReLUMax(6)

        self._layers.append(dw1)
        self._layers.append(bn_dw1)
        self._layers.append(activation)

        if has_se:
            num_reduced_filters = _make_divisible(
                max(1, int(out_channels_dw / 6)), divisor=divisor
            )
            se_block = SEBlock(out_channels_dw, num_reduced_filters, h_swish=h_swish)
            self._layers.append(se_block)

        conv2 = CausalConv1d(
            in_channels=out_channels_dw,
            out_channels=filters,
            kernel_size=1,
            padding=0,
            bias=False,
        )

        bn2 = nn.BatchNorm1d(
            filters,
            eps=1e-3,
            momentum=0.999,
        )

        self._layers.append(conv2)
        self._layers.append(bn2)

        if res and in_channels == filters and stride == 1:
            self.skip_conn = True
            # It serves for the quantization.
            # The behavior remains equivalent for the unquantized models.
            self.op = nnq.FloatFunctional()

    def forward(self, x):
        """Executes the PhiNet convolutional block.

        Arguments
        ---------
        x : torch.Tensor
            Input to the convolutional block.

        Returns
        -------
        torch.Tensor
            Output of the convolutional block.
        """

        if self.skip_conn:
            inp = x

        for layer in self._layers:
            x = layer(x)

        if self.skip_conn:
            return self.op.add(x, inp)  # Equivalent to ``torch.add(a, b)``

        return x

class ResidualUnit(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation,
                 pad_mode="reflect"):
        super(ResidualUnit, self).__init__()

        self.dilaton = dilation

        self.layers = nn.Sequential(
            PhiNetCausalConvBlock(in_channels=in_channels, 
                                  filters=out_channels, 
                                  k_size=7, 
                                  dilation=dilation,
                                  has_se=True,
                                  expansion=1,
                                  stride=1),
            PhiNetCausalConvBlock(in_channels=in_channels, 
                                  filters=out_channels, 
                                  k_size=1, 
                                  has_se=True,
                                  expansion=1,
                                  stride=1),
        )

    def forward(self, x):
        return x + self.layers(x)

class EncoderBlock(nn.Module):
    def __init__(self, out_channels, stride):
        super(EncoderBlock, self).__init__()

        self.layers = nn.Sequential(
            ResidualUnit(
                in_channels=out_channels // 2,
                out_channels=out_channels // 2,
                dilation=1
            ),
            ResidualUnit(
                in_channels=out_channels // 2,
                out_channels=out_channels // 2,
                dilation=3
            ),
            ResidualUnit(
                in_channels=out_channels // 2,
                out_channels=out_channels // 2,
                dilation=9
            ),
            PhiNetCausalConvBlock(
                in_channels=out_channels // 2,
                filters=out_channels,
                k_size=stride*2,
                stride=stride,
                has_se=True,
                expansion=1
            ),
        )

    def forward(self, x):
        return self.layers(x)
    
class Encoder(nn.Module):
    def __init__(self, C, D, strides=(4, 5, 16)):
        super(Encoder, self).__init__()

        self.layers = nn.Sequential(
            CausalConv1d(in_channels=1, out_channels=C, kernel_size=7),
            EncoderBlock(out_channels=2*C, stride=strides[0]),
            EncoderBlock(out_channels=4*C, stride=strides[1]),
            EncoderBlock(out_channels=8*C, stride=strides[2]),
            # EncoderBlock(out_channels=16*C, stride=strides[3]),
            CausalConv1d(in_channels=8*C, out_channels=D, kernel_size=3)
        )

    def forward(self, x):
        return self.layers(x)
    
class EncoderSpec(nn.Module):
    def __init__(self, C, D, n_mel_bins, strides=(4, 5, 16)):
        super(EncoderSpec, self).__init__()

        self.layers = nn.Sequential(
            CausalConv1d(in_channels=n_mel_bins, out_channels=C, kernel_size=7),
            EncoderBlock(out_channels=2*C, stride=strides[0]),
            EncoderBlock(out_channels=4*C, stride=strides[1]),
            EncoderBlock(out_channels=8*C, stride=strides[2]),
            # EncoderBlock(out_channels=16*C, stride=strides[3]),
            CausalConv1d(in_channels=8*C, out_channels=D, kernel_size=3)
        )

    def forward(self, x):
        return self.layers(x)

class PhiSpecNet(nn.Module):
    def __init__(self, C, D, n_mel_bins, strides=(4, 5, 16)):
        super(PhiSpecNet, self).__init__()

        self.layers = nn.Sequential(
            PhiNetCausalConvBlock(
                in_channels=n_mel_bins,
                filters=C,
                k_size=7,
                dilation=1,
                stride=1,
                has_se=True,
                expansion=1
            ),
            PhiNetCausalConvBlock(
                in_channels=C,
                filters=2*C,
                k_size=7,
                stride=2,
                dilation=2,
                has_se=True,
                expansion=1
            ),
            PhiNetCausalConvBlock(
                in_channels=2*C,
                filters=4*C,
                k_size=7,
                stride=2,
                dilation=3,
                has_se=True,
                expansion=1
            ),
            PhiNetCausalConvBlock(
                in_channels=4*C,
                filters=8*C,
                k_size=7,
                stride=3,
                dilation=3,
                has_se=True,
                expansion=1
            ),
            PhiNetCausalConvBlock(
                in_channels=8*C,
                filters=D,
                k_size=3,
                stride=1,
                has_se=True,
                expansion=1
            )
        )

    def forward(self, x):
        return self.layers(x)

class MatchboxNet(nn.Module):
    def __init__(self, input_channels=64, dropout_rate=0.3):
        super(MatchboxNet, self).__init__()
        
        # Conv1 layer - 1 block with kernel=11
        self.conv1 = PhiNetCausalConvBlock(
            in_channels=input_channels,
            filters=64,
            k_size=5,
            stride=2,
            dilation=1,
            has_se=True,
            expansion=1
        )
        self.bn1 = nn.BatchNorm1d(64)  # BatchNorm after conv1
        self.dropout1 = nn.Dropout(dropout_rate)  # Dropout after conv1
        
        # B1 - 2 sub-blocks with kernel=13
        self.b1_1 = PhiNetCausalConvBlock(
            in_channels=64,
            filters=32,
            k_size=3,
            stride=1,
            dilation=2,
            has_se=True,
            expansion=1
        )
        self.bn_b1_1 = nn.BatchNorm1d(32)  # BatchNorm after b1_1
        self.dropout_b1_1 = nn.Dropout(dropout_rate)  # Dropout after b1_1
        
        self.b1_2 = PhiNetCausalConvBlock(
            in_channels=32,
            filters=32,
            k_size=3,
            stride=1,
            dilation=1,
            has_se=True,
            expansion=1
        )
        self.bn_b1_2 = nn.BatchNorm1d(32)  # BatchNorm after b1_2
        self.dropout_b1_2 = nn.Dropout(dropout_rate)  # Dropout after b1_2
        
        # B2 - 2 sub-blocks with kernel=15
        self.b2_1 = PhiNetCausalConvBlock(
            in_channels=32,
            filters=32,
            k_size=3,
            stride=1,
            dilation=4,
            has_se=True,
            expansion=1
        )
        self.bn_b2_1 = nn.BatchNorm1d(32)  # BatchNorm after b2_1
        self.dropout_b2_1 = nn.Dropout(dropout_rate)  # Dropout after b2_1
        
        self.b2_2 = PhiNetCausalConvBlock(
            in_channels=32,
            filters=32,
            k_size=3,
            stride=1,
            dilation=2,
            has_se=True,
            expansion=1
        )
        self.bn_b2_2 = nn.BatchNorm1d(32)  # BatchNorm after b2_2
        self.dropout_b2_2 = nn.Dropout(dropout_rate)  # Dropout after b2_2
        
        # B3 - 2 sub-blocks with kernel=17
        self.b3_1 = PhiNetCausalConvBlock(
            in_channels=32,
            filters=32,
            k_size=3,
            stride=1,
            dilation=1,
            has_se=True,
            expansion=1
        )
        self.bn_b3_1 = nn.BatchNorm1d(32)  # BatchNorm after b3_1
        self.dropout_b3_1 = nn.Dropout(dropout_rate)  # Dropout after b3_1
        
        self.b3_2 = PhiNetCausalConvBlock(
            in_channels=32,
            filters=32,
            k_size=3,
            stride=1,
            dilation=1,
            has_se=True,
            expansion=1
        )
        self.bn_b3_2 = nn.BatchNorm1d(32)  # BatchNorm after b3_2
        self.dropout_b3_2 = nn.Dropout(dropout_rate)  # Dropout after b3_2
        
        # Conv2 layer with kernel=29, dilation=2
        self.conv2 = PhiNetCausalConvBlock(
            in_channels=32,
            filters=64,
            k_size=5,
            stride=1,
            dilation=2,
            has_se=True,
            expansion=1
        )
        self.bn_conv2 = nn.BatchNorm1d(64)  # BatchNorm after conv2
        self.dropout_conv2 = nn.Dropout(dropout_rate)  # Dropout after conv2
        
        # Conv3 layer with kernel=1
        self.conv3 = PhiNetCausalConvBlock(
            in_channels=64,
            filters=64,
            k_size=1,
            stride=1,
            dilation=1,
            has_se=True,
            expansion=1
        )
        self.bn_conv3 = nn.BatchNorm1d(64)  # BatchNorm after conv3
        self.dropout_conv3 = nn.Dropout(dropout_rate)  # Dropout after conv3
        
    def forward(self, x):
        # Input shape: [batch_size, input_channels, time]
        
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        
        # B1 block
        x = self.b1_1(x)
        x = self.bn_b1_1(x)
        x = self.dropout_b1_1(x)
        
        x = self.b1_2(x)
        x = self.bn_b1_2(x)
        x = self.dropout_b1_2(x)
        
        # B2 block
        x = self.b2_1(x)
        x = self.bn_b2_1(x)
        x = self.dropout_b2_1(x)
        
        x = self.b2_2(x)
        x = self.bn_b2_2(x)
        x = self.dropout_b2_2(x)
        
        # B3 block
        x = self.b3_1(x)
        x = self.bn_b3_1(x)
        x = self.dropout_b3_1(x)
        
        x = self.b3_2(x)
        x = self.bn_b3_2(x)
        x = self.dropout_b3_2(x)
        
        # Final convolutions
        x = self.conv2(x)
        x = self.bn_conv2(x)
        x = self.dropout_conv2(x)
        
        x = self.conv3(x)
        x = self.bn_conv3(x)
        x = self.dropout_conv3(x)
        
        return x

class MatchboxNetSkip(nn.Module):
    def __init__(self, input_channels=64, dropout_rate=0.3):
        super(MatchboxNetSkip, self).__init__()
        
        # Conv1 layer - 1 block with kernel=11
        self.conv1 = PhiNetCausalConvBlock(
            in_channels=input_channels,
            filters=64,
            k_size=5,
            stride=1,  # Changed from 2 to preserve temporal resolution
            dilation=1,
            has_se=True,
            expansion=1
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # B1 - 2 sub-blocks with skip connections
        self.b1_1 = PhiNetCausalConvBlock(
            in_channels=64,
            filters=32,
            k_size=3,
            stride=1,
            dilation=2,
            has_se=True,
            expansion=1
        )
        self.bn_b1_1 = nn.BatchNorm1d(32)
        self.dropout_b1_1 = nn.Dropout(dropout_rate)
        
        # Projection for first skip connection (64->32 channels)
        self.proj1 = nn.Conv1d(64, 32, kernel_size=1, stride=1)
        
        self.b1_2 = PhiNetCausalConvBlock(
            in_channels=32,
            filters=32,
            k_size=3,
            stride=1,
            dilation=1,
            has_se=True,
            expansion=1
        )
        self.bn_b1_2 = nn.BatchNorm1d(32)
        self.dropout_b1_2 = nn.Dropout(dropout_rate)
        
        # B2 - 2 sub-blocks with skip connections
        self.b2_1 = PhiNetCausalConvBlock(
            in_channels=32,
            filters=32,
            k_size=3,
            stride=1,
            dilation=4,
            has_se=True,
            expansion=1
        )
        self.bn_b2_1 = nn.BatchNorm1d(32)
        self.dropout_b2_1 = nn.Dropout(dropout_rate)
        
        self.b2_2 = PhiNetCausalConvBlock(
            in_channels=32,
            filters=32,
            k_size=3,
            stride=1,
            dilation=2,
            has_se=True,
            expansion=1
        )
        self.bn_b2_2 = nn.BatchNorm1d(32)
        self.dropout_b2_2 = nn.Dropout(dropout_rate)
        
        # B3 - 2 sub-blocks with skip connections
        self.b3_1 = PhiNetCausalConvBlock(
            in_channels=32,
            filters=32,
            k_size=3,
            stride=1,
            dilation=1,
            has_se=True,
            expansion=1
        )
        self.bn_b3_1 = nn.BatchNorm1d(32)
        self.dropout_b3_1 = nn.Dropout(dropout_rate)
        
        self.b3_2 = PhiNetCausalConvBlock(
            in_channels=32,
            filters=32,
            k_size=3,
            stride=1,
            dilation=1,
            has_se=True,
            expansion=1
        )
        self.bn_b3_2 = nn.BatchNorm1d(32)
        self.dropout_b3_2 = nn.Dropout(dropout_rate)
        
        # Projection for final skip connection (32->64 channels)
        self.proj2 = nn.Conv1d(32, 64, kernel_size=1, stride=1)
        
        # Conv2 layer with kernel=29, dilation=2
        self.conv2 = PhiNetCausalConvBlock(
            in_channels=32,
            filters=64,
            k_size=5,
            stride=1,
            dilation=2,
            has_se=True,
            expansion=1
        )
        self.bn_conv2 = nn.BatchNorm1d(64)
        self.dropout_conv2 = nn.Dropout(dropout_rate)
        
        # Conv3 layer with kernel=1
        self.conv3 = PhiNetCausalConvBlock(
            in_channels=64,
            filters=64,
            k_size=1,
            stride=1,
            dilation=1,
            has_se=True,
            expansion=1
        )
        self.bn_conv3 = nn.BatchNorm1d(64)
        self.dropout_conv3 = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # Input shape: [batch_size, input_channels, time]
        
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Store the output for skip connection
        conv1_out = x
        
        # B1 block with skip connections
        # First sub-block
        x = self.b1_1(x)
        x = self.bn_b1_1(x)
        x = F.relu(x)
        x = self.dropout_b1_1(x)
        
        # Add skip connection from conv1 (with projection)
        projected_conv1 = self.proj1(conv1_out)
        # Make sure shapes match before adding
        if x.shape[2] != projected_conv1.shape[2]:
            # Adjust temporal dimension if needed
            diff = abs(x.shape[2] - projected_conv1.shape[2])
            if x.shape[2] > projected_conv1.shape[2]:
                projected_conv1 = F.pad(projected_conv1, (0, diff))
            else:
                x = F.pad(x, (0, diff))
        x = x + projected_conv1
        
        # Store for next skip connection
        b1_1_out = x
        
        # Second sub-block
        x = self.b1_2(x)
        x = self.bn_b1_2(x)
        x = F.relu(x)
        x = self.dropout_b1_2(x)
        
        # Add skip connection from first sub-block
        if x.shape[2] != b1_1_out.shape[2]:
            diff = abs(x.shape[2] - b1_1_out.shape[2])
            if x.shape[2] > b1_1_out.shape[2]:
                b1_1_out = F.pad(b1_1_out, (0, diff))
            else:
                x = F.pad(x, (0, diff))
        x = x + b1_1_out
        
        # Store for block skip connection
        b1_out = x
        
        # B2 block with skip connections
        # First sub-block
        x = self.b2_1(x)
        x = self.bn_b2_1(x)
        x = F.relu(x)
        x = self.dropout_b2_1(x)
        
        # Add skip connection from B1 output
        if x.shape[2] != b1_out.shape[2]:
            diff = abs(x.shape[2] - b1_out.shape[2])
            if x.shape[2] > b1_out.shape[2]:
                b1_out = F.pad(b1_out, (0, diff))
            else:
                x = F.pad(x, (0, diff))
        x = x + b1_out
        
        # Store for next skip connection
        b2_1_out = x
        
        # Second sub-block
        x = self.b2_2(x)
        x = self.bn_b2_2(x)
        x = F.relu(x)
        x = self.dropout_b2_2(x)
        
        # Add skip connection from first sub-block
        if x.shape[2] != b2_1_out.shape[2]:
            diff = abs(x.shape[2] - b2_1_out.shape[2])
            if x.shape[2] > b2_1_out.shape[2]:
                b2_1_out = F.pad(b2_1_out, (0, diff))
            else:
                x = F.pad(x, (0, diff))
        x = x + b2_1_out
        
        # Store for block skip connection
        b2_out = x
        
        # B3 block with skip connections
        # First sub-block
        x = self.b3_1(x)
        x = self.bn_b3_1(x)
        x = F.relu(x)
        x = self.dropout_b3_1(x)
        
        # Add skip connection from B2 output
        if x.shape[2] != b2_out.shape[2]:
            diff = abs(x.shape[2] - b2_out.shape[2])
            if x.shape[2] > b2_out.shape[2]:
                b2_out = F.pad(b2_out, (0, diff))
            else:
                x = F.pad(x, (0, diff))
        x = x + b2_out
        
        # Store for next skip connection
        b3_1_out = x
        
        # Second sub-block
        x = self.b3_2(x)
        x = self.bn_b3_2(x)
        x = F.relu(x)
        x = self.dropout_b3_2(x)
        
        # Add skip connection from first sub-block
        if x.shape[2] != b3_1_out.shape[2]:
            diff = abs(x.shape[2] - b3_1_out.shape[2])
            if x.shape[2] > b3_1_out.shape[2]:
                b3_1_out = F.pad(b3_1_out, (0, diff))
            else:
                x = F.pad(x, (0, diff))
        x = x + b3_1_out
        
        # Store final block output
        b3_out = x
        
        # Final convolutions with skip connection
        x = self.conv2(x)
        x = self.bn_conv2(x)
        x = F.relu(x)
        x = self.dropout_conv2(x)
        
        # Project b3_out for skip connection (32->64 channels)
        projected_b3 = self.proj2(b3_out)
        
        # Add skip connection from B3 output
        if x.shape[2] != projected_b3.shape[2]:
            diff = abs(x.shape[2] - projected_b3.shape[2])
            if x.shape[2] > projected_b3.shape[2]:
                projected_b3 = F.pad(projected_b3, (0, diff))
            else:
                x = F.pad(x, (0, diff))
        x = x + projected_b3
        
        # Store for final skip connection
        conv2_out = x
        
        # Final 1x1 convolution
        x = self.conv3(x)
        x = self.bn_conv3(x)
        x = F.relu(x)
        x = self.dropout_conv3(x)
        
        # Final skip connection
        if x.shape[2] != conv2_out.shape[2]:
            diff = abs(x.shape[2] - conv2_out.shape[2])
            if x.shape[2] > conv2_out.shape[2]:
                conv2_out = F.pad(conv2_out, (0, diff))
            else:
                x = F.pad(x, (0, diff))
        x = x + conv2_out
        
        return x    

class SRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout_rate=0.3):
        super(SRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        # Input and recurrent weights
        self.Wx = nn.Linear(input_size, hidden_size)  # Input weights
        self.Wh = nn.Linear(hidden_size, hidden_size, bias=False)  # Recurrent weights

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Persistent hidden state
        self.h_t = None

    def reset_hidden_state(self, batch_size, device):
        """Reset the hidden state to zeros."""
        self.h_t = torch.zeros(batch_size, self.hidden_size, device=device)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]
        batch_size, seq_len, input_size = x.shape

        # Initialize hidden state if not already set or if batch size has changed
        if self.h_t is None or self.h_t.size(0) != batch_size:
            self.reset_hidden_state(batch_size, x.device)

        # Process each time step in the chunk
        for t in range(seq_len):
            # Compute Wx(x[:, t, :]) -> [batch_size, hidden_size]
            Wx_out = self.Wx(x[:, t, :])
            
            # Compute Wh(h_t) -> [batch_size, hidden_size]
            Wh_out = self.Wh(self.h_t)
            
            # Update hidden state with tanh activation
            self.h_t = torch.tanh(Wx_out + Wh_out)
            
            # Apply dropout to the hidden state
            self.h_t = self.dropout(self.h_t)

        return self.h_t  # Return the updated hidden state


class HighwayGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False, dropout=0.0, bidirectional=False):
        super(HighwayGRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        # GRU gates: update and reset gates, candidate hidden state
        self.update_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.candidate = nn.Linear(input_size + hidden_size, hidden_size)
        
        # Highway gate to decide how much of the hidden state should be passed through
        self.highway_gate = nn.Linear(input_size + hidden_size, hidden_size)
        
        # Layer normalizations for stability
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Output layer (optional if you want to use it like the GRU)
        self.fc = nn.Linear(hidden_size, hidden_size) 

        # Dropout layer
        self.drop = nn.Dropout(dropout)

    def forward(self, x, h=None):
        # If batch_first is True, we need to reshape x accordingly
        if self.batch_first:
            x = x.permute(1, 0, 2)  # Convert to (seq_len, batch, input_size)

        seq_len, batch_size, _ = x.size()
        
        if h is None:
            h = torch.zeros(self.num_layers * (1 + self.bidirectional), batch_size, self.hidden_size, device=x.device)

        # Prepare for the output (hidden states over time)
        all_h = []

        for t in range(seq_len):
            x_t = x[t, :, :]  # Get input at time step t
            combined = torch.cat((x_t, h[-1]), dim=1)

            # Compute GRU gates
            z_t = torch.sigmoid(self.update_gate(combined))  # Update gate
            r_t = torch.sigmoid(self.reset_gate(combined))  # Reset gate

            # Compute candidate hidden state
            h_tilde = torch.tanh(self.candidate(torch.cat((x_t, r_t * h[-1]), dim=1)))

            # Compute final hidden state
            h_new = (1 - z_t) * h[-1] + z_t * h_tilde

            # Highway gate
            highway = torch.sigmoid(self.highway_gate(combined))  # Highway gate
            h_final = highway * h_new + (1 - highway) * h[-1]  # Combine GRU state with highway gate

            # Apply dropout if needed
            h_final = self.drop(h_final)

            # Optionally apply layer normalization
            h_final = self.layer_norm(h_final)

            # Store hidden states
            all_h.append(h_final.unsqueeze(0))

            # Update h for the next time step
            h = h_new.unsqueeze(0)

        # Combine hidden states over all time steps
        all_h = torch.cat(all_h, dim=0)

        if self.batch_first:
            all_h = all_h.permute(1, 0, 2)  # Convert back to (batch_size, seq_len, hidden_size)

        return all_h, h
    


if __name__ == '__main__':
    C = 8
    D = 64
    n_mel_bins = 64
    strides = (2, 2, 3)
    encoder = MatchboxNetSkip(input_channels=64).cuda()
    summary(encoder, (n_mel_bins, 101))