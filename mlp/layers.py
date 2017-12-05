# -*- coding: utf-8 -*-
"""Layer definitions.

This module defines classes which encapsulate a single layer.

These layers map input activations to output activation with the `fprop`
method and map gradients with repsect to outputs to gradients with respect to
their inputs with the `bprop` method.

Some layers will have learnable parameters and so will additionally define
methods for getting and setting parameter and calculating gradients with
respect to the layer parameters.
"""

import numpy as np
import mlp.initialisers as init
from mlp import DEFAULT_SEED
from scipy import signal

class Layer(object):
    """Abstract class defining the interface for a layer."""

    def fprop(self, inputs):
        """Forward propagates activations through the layer transformation.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).

        Returns:
            outputs: Array of layer outputs of shape (batch_size, output_dim).
        """
        raise NotImplementedError()

    def bprop(self, inputs, outputs, grads_wrt_outputs):
        """Back propagates gradients through a layer.

        Given gradients with respect to the outputs of the layer calculates the
        gradients with respect to the layer inputs.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).
            outputs: Array of layer outputs calculated in forward pass of
                shape (batch_size, output_dim).
            grads_wrt_outputs: Array of gradients with respect to the layer
                outputs of shape (batch_size, output_dim).

        Returns:
            Array of gradients with respect to the layer inputs of shape
            (batch_size, input_dim).
        """
        raise NotImplementedError()


class LayerWithParameters(Layer):
    """Abstract class defining the interface for a layer with parameters."""

    def grads_wrt_params(self, inputs, grads_wrt_outputs):
        """Calculates gradients with respect to layer parameters.

        Args:
            inputs: Array of inputs to layer of shape (batch_size, input_dim).
            grads_wrt_to_outputs: Array of gradients with respect to the layer
                outputs of shape (batch_size, output_dim).

        Returns:
            List of arrays of gradients with respect to the layer parameters
            with parameter gradients appearing in same order in tuple as
            returned from `get_params` method.
        """
        raise NotImplementedError()

    def params_penalty(self):
        """Returns the parameter dependent penalty term for this layer.

        If no parameter-dependent penalty terms are set this returns zero.
        """
        raise NotImplementedError()

    @property
    def params(self):
        """Returns a list of parameters of layer.

        Returns:
            List of current parameter values. This list should be in the
            corresponding order to the `values` argument to `set_params`.
        """
        raise NotImplementedError()

    @params.setter
    def params(self, values):
        """Sets layer parameters from a list of values.

        Args:
            values: List of values to set parameters to. This list should be
                in the corresponding order to what is returned by `get_params`.
        """
        raise NotImplementedError()

class StochasticLayerWithParameters(Layer):
    """Specialised layer which uses a stochastic forward propagation."""

    def __init__(self, rng=None):
        """Constructs a new StochasticLayer object.

        Args:
            rng (RandomState): Seeded random number generator object.
        """
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng

    def fprop(self, inputs, stochastic=True):
        """Forward propagates activations through the layer transformation.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).
            stochastic: Flag allowing different deterministic
                forward-propagation mode in addition to default stochastic
                forward-propagation e.g. for use at test time. If False
                a deterministic forward-propagation transformation
                corresponding to the expected output of the stochastic
                forward-propagation is applied.

        Returns:
            outputs: Array of layer outputs of shape (batch_size, output_dim).
        """
        raise NotImplementedError()
    def grads_wrt_params(self, inputs, grads_wrt_outputs):
        """Calculates gradients with respect to layer parameters.

        Args:
            inputs: Array of inputs to layer of shape (batch_size, input_dim).
            grads_wrt_to_outputs: Array of gradients with respect to the layer
                outputs of shape (batch_size, output_dim).

        Returns:
            List of arrays of gradients with respect to the layer parameters
            with parameter gradients appearing in same order in tuple as
            returned from `get_params` method.
        """
        raise NotImplementedError()

    def params_penalty(self):
        """Returns the parameter dependent penalty term for this layer.

        If no parameter-dependent penalty terms are set this returns zero.
        """
        raise NotImplementedError()

    @property
    def params(self):
        """Returns a list of parameters of layer.

        Returns:
            List of current parameter values. This list should be in the
            corresponding order to the `values` argument to `set_params`.
        """
        raise NotImplementedError()

    @params.setter
    def params(self, values):
        """Sets layer parameters from a list of values.

        Args:
            values: List of values to set parameters to. This list should be
                in the corresponding order to what is returned by `get_params`.
        """
        raise NotImplementedError()

class StochasticLayer(Layer):
    """Specialised layer which uses a stochastic forward propagation."""

    def __init__(self, rng=None):
        """Constructs a new StochasticLayer object.

        Args:
            rng (RandomState): Seeded random number generator object.
        """
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng

    def fprop(self, inputs, stochastic=True):
        """Forward propagates activations through the layer transformation.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).
            stochastic: Flag allowing different deterministic
                forward-propagation mode in addition to default stochastic
                forward-propagation e.g. for use at test time. If False
                a deterministic forward-propagation transformation
                corresponding to the expected output of the stochastic
                forward-propagation is applied.

        Returns:
            outputs: Array of layer outputs of shape (batch_size, output_dim).
        """
        raise NotImplementedError()

    def bprop(self, inputs, outputs, grads_wrt_outputs):
        """Back propagates gradients through a layer.

        Given gradients with respect to the outputs of the layer calculates the
        gradients with respect to the layer inputs. This should correspond to
        default stochastic forward-propagation.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).
            outputs: Array of layer outputs calculated in forward pass of
                shape (batch_size, output_dim).
            grads_wrt_outputs: Array of gradients with respect to the layer
                outputs of shape (batch_size, output_dim).

        Returns:
            Array of gradients with respect to the layer inputs of shape
            (batch_size, input_dim).
        """
        raise NotImplementedError()


class AffineLayer(LayerWithParameters):
    """Layer implementing an affine tranformation of its inputs.

    This layer is parameterised by a weight matrix and bias vector.
    """

    def __init__(self, input_dim, output_dim,
                 weights_initialiser=init.UniformInit(-0.1, 0.1),
                 biases_initialiser=init.ConstantInit(0.),
                 weights_penalty=None, biases_penalty=None):
        """Initialises a parameterised affine layer.

        Args:
            input_dim (int): Dimension of inputs to the layer.
            output_dim (int): Dimension of the layer outputs.
            weights_initialiser: Initialiser for the weight parameters.
            biases_initialiser: Initialiser for the bias parameters.
            weights_penalty: Weights-dependent penalty term (regulariser) or
                None if no regularisation is to be applied to the weights.
            biases_penalty: Biases-dependent penalty term (regulariser) or
                None if no regularisation is to be applied to the biases.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = weights_initialiser((self.output_dim, self.input_dim))
        self.biases = biases_initialiser(self.output_dim)
        self.weights_penalty = weights_penalty
        self.biases_penalty = biases_penalty

    def fprop(self, inputs):
        """Forward propagates activations through the layer transformation.

        For inputs `x`, outputs `y`, weights `W` and biases `b` the layer
        corresponds to `y = W.dot(x) + b`.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).

        Returns:
            outputs: Array of layer outputs of shape (batch_size, output_dim).
        """
        return self.weights.dot(inputs.T).T + self.biases

    def bprop(self, inputs, outputs, grads_wrt_outputs):
        """Back propagates gradients through a layer.

        Given gradients with respect to the outputs of the layer calculates the
        gradients with respect to the layer inputs.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).
            outputs: Array of layer outputs calculated in forward pass of
                shape (batch_size, output_dim).
            grads_wrt_outputs: Array of gradients with respect to the layer
                outputs of shape (batch_size, output_dim).

        Returns:
            Array of gradients with respect to the layer inputs of shape
            (batch_size, input_dim).
        """
        return grads_wrt_outputs.dot(self.weights)

    def grads_wrt_params(self, inputs, grads_wrt_outputs):
        """Calculates gradients with respect to layer parameters.

        Args:
            inputs: array of inputs to layer of shape (batch_size, input_dim)
            grads_wrt_to_outputs: array of gradients with respect to the layer
                outputs of shape (batch_size, output_dim)

        Returns:
            list of arrays of gradients with respect to the layer parameters
            `[grads_wrt_weights, grads_wrt_biases]`.
        """

        grads_wrt_weights = np.dot(grads_wrt_outputs.T, inputs)
        grads_wrt_biases = np.sum(grads_wrt_outputs, axis=0)

        if self.weights_penalty is not None:
            grads_wrt_weights += self.weights_penalty.grad(self.weights)

        if self.biases_penalty is not None:
            grads_wrt_biases += self.biases_penalty.grad(self.biases)

        return [grads_wrt_weights, grads_wrt_biases]

    def params_penalty(self):
        """Returns the parameter dependent penalty term for this layer.

        If no parameter-dependent penalty terms are set this returns zero.
        """
        params_penalty = 0
        if self.weights_penalty is not None:
            params_penalty += self.weights_penalty(self.weights)
        if self.biases_penalty is not None:
            params_penalty += self.biases_penalty(self.biases)
        return params_penalty

    @property
    def params(self):
        """A list of layer parameter values: `[weights, biases]`."""
        return [self.weights, self.biases]

    @params.setter
    def params(self, values):
        self.weights = values[0]
        self.biases = values[1]

    def __repr__(self):
        return 'AffineLayer(input_dim={0}, output_dim={1})'.format(
            self.input_dim, self.output_dim)

class BatchNormalizationLayer(StochasticLayerWithParameters):
    """Layer implementing an affine tranformation of its inputs.

    This layer is parameterised by a weight matrix and bias vector.
    """

    def __init__(self, input_dim, rng=None):
        """Initialises a parameterised affine layer.

        Args:
            input_dim (int): Dimension of inputs to the layer.
            output_dim (int): Dimension of the layer outputs.
            weights_initialiser: Initialiser for the weight parameters.
            biases_initialiser: Initialiser for the bias parameters.
            weights_penalty: Weights-dependent penalty term (regulariser) or
                None if no regularisation is to be applied to the weights.
            biases_penalty: Biases-dependent penalty term (regulariser) or
                None if no regularisation is to be applied to the biases.
        """
        super(BatchNormalizationLayer, self).__init__(rng)
        self.beta = np.random.normal(size=(input_dim))
        self.gamma = np.random.normal(size=(input_dim))
        self.epsilon = 0.00001
        self.cache = None
        self.input_dim = input_dim

    def fprop(self, inputs, stochastic=True):
        """Forward propagates inputs through a layer."""
        mu = np.mean(inputs, axis = 0)
        xmu = inputs - mu
        var = np.mean(xmu**2, axis = 0)
        sqrtvar = np.sqrt(var + self.epsilon)
        ivar = 1./sqrtvar
        xhat = xmu / sqrtvar
        self.cache = (xmu, var, xhat, sqrtvar, ivar)
        return self.gamma * xhat + self.beta
        # raise NotImplementedError

    def bprop(self, inputs, outputs, grads_wrt_outputs):
        """Back propagates gradients through a layer.

        Given gradients with respect to the outputs of the layer calculates the
        gradients with respect to the layer inputs.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).
            outputs: Array of layer outputs calculated in forward pass of
                shape (batch_size, output_dim).
            grads_wrt_outputs: Array of gradients with respect to the layer
                outputs of shape (batch_size, output_dim).

        Returns:
            Array of gradients with respect to the layer inputs of shape
            (batch_size, input_dim).
        """
        xmu, var, xhat, sqrtvar, ivar = self.cache
        dxhat = grads_wrt_outputs * self.gamma
        divar = np.sum(dxhat * xmu, axis=0)
        dxmu1 = dxhat * ivar
        dsqrtvar = divar * (-1. / sqrtvar**2)
        dvar = 0.5 * 1 / np.sqrt(var + self.epsilon) * dsqrtvar
        dsq = 1. / outputs.shape[0] * np.ones_like(outputs) * dvar
        dxmu2 = 2 * xmu * dsq
        dx1 = dxmu1 + dxmu2
        dmu = -1. * np.sum(dxmu1 + dxmu2, axis=0)
        dx2 = 1. / outputs.shape[0] *  np.ones_like(outputs)* dmu
        dx = dx1 + dx2

        return dx
        # raise NotImplementedError

    def grads_wrt_params(self, inputs, grads_wrt_outputs):
        """Calculates gradients with respect to layer parameters.

        Args:
            inputs: array of inputs to layer of shape (batch_size, input_dim)
            grads_wrt_to_outputs: array of gradients with respect to the layer
                outputs of shape (batch_size, output_dim)

        Returns:
            list of arrays of gradients with respect to the layer parameters
            `[grads_wrt_weights, grads_wrt_biases]`.
        """
        xmu, var, xhat, sqrtvar, ivar = self.cache
        dgamma = np.sum(grads_wrt_outputs * xhat, axis=0)
        dbeta =  grads_wrt_outputs.sum(axis=0)
        return [dgamma, dbeta]
        # raise NotImplementedError

    def params_penalty(self):
        """Returns the parameter dependent penalty term for this layer.

        If no parameter-dependent penalty terms are set this returns zero.
        """
        params_penalty = 0

        return params_penalty

    @property
    def params(self):
        """A list of layer parameter values: `[gammas, betas]`."""
        return [self.gamma, self.beta]

    @params.setter
    def params(self, values):
        self.gamma = values[0]
        self.beta = values[1]

    def __repr__(self):
        return 'BatchNormalizationLayer(input_dim={0})'.format(
            self.input_dim)


class SigmoidLayer(Layer):
    """Layer implementing an element-wise logistic sigmoid transformation."""

    def fprop(self, inputs):
        """Forward propagates activations through the layer transformation.

        For inputs `x` and outputs `y` this corresponds to
        `y = 1 / (1 + exp(-x))`.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).

        Returns:
            outputs: Array of layer outputs of shape (batch_size, output_dim).
        """
        return 1. / (1. + np.exp(-inputs))

    def bprop(self, inputs, outputs, grads_wrt_outputs):
        """Back propagates gradients through a layer.

        Given gradients with respect to the outputs of the layer calculates the
        gradients with respect to the layer inputs.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).
            outputs: Array of layer outputs calculated in forward pass of
                shape (batch_size, output_dim).
            grads_wrt_outputs: Array of gradients with respect to the layer
                outputs of shape (batch_size, output_dim).

        Returns:
            Array of gradients with respect to the layer inputs of shape
            (batch_size, input_dim).
        """
        return grads_wrt_outputs * outputs * (1. - outputs)

    def __repr__(self):
        return 'SigmoidLayer'

class ConvolutionalLayer(LayerWithParameters):
    """Layer implementing a 2D convolution-based transformation of its inputs.
    The layer is parameterised by a set of 2D convolutional kernels, a four
    dimensional array of shape
        (num_output_channels, num_input_channels, kernel_dim_1, kernel_dim_2)
    and a bias vector, a one dimensional array of shape
        (num_output_channels,)
    i.e. one shared bias per output channel.
    Assuming no-padding is applied to the inputs so that outputs are only
    calculated for positions where the kernel filters fully overlap with the
    inputs, and that unit strides are used the outputs will have spatial extent
        output_dim_1 = input_dim_1 - kernel_dim_1 + 1
        output_dim_2 = input_dim_2 - kernel_dim_2 + 1
    """

    def __init__(self, num_input_channels, num_output_channels,
                 input_dim_1, input_dim_2,
                 kernel_dim_1, kernel_dim_2,
                 kernels_init=init.UniformInit(-0.01, 0.01),
                 biases_init=init.ConstantInit(0.),
                 kernels_penalty=None, biases_penalty=None):
        """Initialises a parameterised convolutional layer.
        Args:
            num_input_channels (int): Number of channels in inputs to
                layer (this may be number of colour channels in the input
                images if used as the first layer in a model, or the
                number of output channels, a.k.a. feature maps, from a
                a previous convolutional layer).
            num_output_channels (int): Number of channels in outputs
                from the layer, a.k.a. number of feature maps.
            input_dim_1 (int): Size of first input dimension of each 2D
                channel of inputs.
            input_dim_2 (int): Size of second input dimension of each 2D
                channel of inputs.
            kernel_dim_1 (int): Size of first dimension of each 2D channel of
                kernels.
            kernel_dim_2 (int): Size of second dimension of each 2D channel of
                kernels.
            kernels_intialiser: Initialiser for the kernel parameters.
            biases_initialiser: Initialiser for the bias parameters.
            kernels_penalty: Kernel-dependent penalty term (regulariser) or
                None if no regularisation is to be applied to the kernels.
            biases_penalty: Biases-dependent penalty term (regulariser) or
                None if no regularisation is to be applied to the biases.
        """
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.input_dim_1 = input_dim_1
        self.input_dim_2 = input_dim_2
        self.kernel_dim_1 = kernel_dim_1
        self.kernel_dim_2 = kernel_dim_2
        self.kernels_init = kernels_init
        self.biases_init = biases_init
        self.kernels_shape = (
            num_output_channels, num_input_channels, kernel_dim_1, kernel_dim_2
        )
        self.inputs_shape = (
            None, num_input_channels, input_dim_1, input_dim_2
        )
        self.kernels = self.kernels_init(self.kernels_shape)
        self.biases = self.biases_init(num_output_channels)
        self.kernels_penalty = kernels_penalty
        self.biases_penalty = biases_penalty


    def fprop(self, inputs):
        """Forward propagates activations through the layer transformation.
        For inputs `x`, outputs `y`, kernels `K` and biases `b` the layer
        corresponds to `y = conv2d(x, K) + b`.
        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).
        Returns:
            outputs: Array of layer outputs of shape (batch_size, output_dim).
        """
        FN, C, FH, FW = self.kernels_shape
        N, C, H, W  = inputs.shape
        out_h = int(1 + H - FH)
        out_w = int(1 + W - FW)
        out = np.zeros((N, FN, out_h, out_w))
        for n in range(N):
            for fn in range(FN):
                for c in range(C):
                    out[n,fn] += signal.convolve2d(inputs[n,c], self.kernels[fn,c], 'valid')
                out[n,fn] += self.biases[fn]
        return out
        # raise NotImplementedError
        # c = c
        # fn = k

    def bprop(self, inputs, outputs, grads_wrt_outputs):
        """Back propagates gradients through a layer.
        Given gradients with respect to the outputs of the layer calculates the
        gradients with respect to the layer inputs.
        Args:
            inputs: Array of layer inputs of shape
                (batch_size, num_input_channels, input_dim_1, input_dim_2).
            outputs: Array of layer outputs calculated in forward pass of
                shape
                (batch_size, num_output_channels, output_dim_1, output_dim_2).
            grads_wrt_outputs: Array of gradients with respect to the layer
                outputs of shape
                (batch_size, num_output_channels, output_dim_1, output_dim_2).
        Returns:
            Array of gradients with respect to the layer inputs of shape
            (batch_size, input_dim).
        """
        # Pad the grads_wrt_outputs
        FN, C, FH, FW = self.kernels.shape
        N, FN, out_h, out_w = outputs.shape
        N, C, H, W = inputs.shape
        dx = np.zeros((N, C, H, W))
        for n in range(N):
            for c in range(C):
                for fn in range(FN):
                    dx[n,c] += signal.convolve2d(grads_wrt_outputs[n,fn], np.rot90(self.kernels[fn,c], 2))

                    # I don't know why but the codes below does not work
                    # for i in range(H):
                    #     for j in range(W):
                    #         for s in range(FH):
                    #             for t in range(FW): #
                    #                 if i - t >= 0 and i - t < out_h and j - s >= 0 and j - s < out_w:
                    #                     dx[n, c, i, j] += grads_wrt_outputs[n, fn, i - t, j - s] * self.kernels[fn, c, s, t]
                    #                     print(dx[n, c, i, j])
                    #                     print(fn, n, c, i, j, s, t)

        return dx
        # raise NotImplementedError

    def grads_wrt_params(self, inputs, grads_wrt_outputs):
        """Calculates gradients with respect to layer parameters.
        Args:
            inputs: array of inputs to layer of shape (batch_size, input_dim)
            grads_wrt_to_outputs: array of gradients with respect to the layer
                outputs of shape
                (batch_size, num_output-_channels, output_dim_1, output_dim_2).
        Returns:
            list of arrays of gradients with respect to the layer parameters
            `[grads_wrt_kernels, grads_wrt_biases]`.
        """

        FN, C, FH, FW = self.kernels.shape
        N, C, H, W = inputs.shape
        out_h = (H - FH) + 1
        out_w = (W - FW) + 1
        #grad (N, FN, out_h, out_w)
        db = np.einsum('ijkl -> j', grads_wrt_outputs)
        dW = np.zeros((FN, C, FH, FW))
        for n in range(N):
            for fn in range(FN):
                for c in range(C):
                    dW[fn,c] += np.rot90(signal.convolve2d(inputs[n,c] , np.rot90(grads_wrt_outputs[n,fn], 2) , 'valid'),2)
                    # for i in range(out_h):
                    #     for j in range(out_w):
                    #         for s in range(FH):
                    #             for t in range(FW):
                    #                 dW[fn, c, FW - t - 1, FH - s - 1] += inputs[n, c, i + t, j + s] * grads_wrt_outputs[n, fn, i, j]
        if self.kernels_penalty is not None:
            dW += self.kernels_penalty.grad(self.kernels)

        if self.biases_penalty is not None:
            db += self.biases_penalty.grad(self.biases)

        return [dW ,db]

        # raise NotImplementedError

    def params_penalty(self):
        """Returns the parameter dependent penalty term for this layer.
        If no parameter-dependent penalty terms are set this returns zero.
        """
        params_penalty = 0
        if self.kernels_penalty is not None:
            params_penalty += self.kernels_penalty(self.kernels)
        if self.biases_penalty is not None:
            params_penalty += self.biases_penalty(self.biases)
        return params_penalty

    @property
    def params(self):
        """A list of layer parameter values: `[kernels, biases]`."""
        return [self.kernels, self.biases]

    @params.setter
    def params(self, values):
        self.kernels = values[0]
        self.biases = values[1]

    def __repr__(self):
        return (
            'ConvolutionalLayer(\n'
            '    num_input_channels={0}, num_output_channels={1},\n'
            '    input_dim_1={2}, input_dim_2={3},\n'
            '    kernel_dim_1={4}, kernel_dim_2={5}\n'
            ')'
            .format(self.num_input_channels, self.num_output_channels,
                    self.input_dim_1, self.input_dim_2, self.kernel_dim_1,
                    self.kernel_dim_2)
        )


class ReluLayer(Layer):
    """Layer implementing an element-wise rectified linear transformation."""

    def fprop(self, inputs):
        """Forward propagates activations through the layer transformation.

        For inputs `x` and outputs `y` this corresponds to `y = max(0, x)`.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).

        Returns:
            outputs: Array of layer outputs of shape (batch_size, output_dim).
        """
        # print(inputs.shape)
        return np.maximum(inputs, 0.)

    def bprop(self, inputs, outputs, grads_wrt_outputs):
        """Back propagates gradients through a layer.

        Given gradients with respect to the outputs of the layer calculates the
        gradients with respect to the layer inputs.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).
            outputs: Array of layer outputs calculated in forward pass of
                shape (batch_size, output_dim).
            grads_wrt_outputs: Array of gradients with respect to the layer
                outputs of shape (batch_size, output_dim).

        Returns:
            Array of gradients with respect to the layer inputs of shape
            (batch_size, input_dim).
        """
        return (outputs > 0) * grads_wrt_outputs

    def __repr__(self):
        return 'ReluLayer'

class LeakyReluLayer(Layer):
    """Layer implementing an element-wise rectified linear transformation."""
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def fprop(self, inputs):
        """Forward propagates activations through the layer transformation.

        For inputs `x` and outputs `y` this corresponds to `y = max(0, x)`.
        """
        positive_inputs = np.maximum(inputs, 0.)

        negative_inputs = inputs
        negative_inputs[negative_inputs>0] = 0.
        negative_inputs = negative_inputs * self.alpha

        outputs = positive_inputs + negative_inputs
        return outputs

    def bprop(self, inputs, outputs, grads_wrt_outputs):
        """Back propagates gradients through a layer.

        Given gradients with respect to the outputs of the layer calculates the
        gradients with respect to the layer inputs.
        """
        positive_gradients = (outputs > 0) * grads_wrt_outputs
        negative_gradients = self.alpha * (outputs < 0) * grads_wrt_outputs
        gradients = positive_gradients + negative_gradients
        return gradients

    def __repr__(self):
        return 'LeakyReluLayer'

class ELULayer(Layer):
    """Layer implementing an ELU activation."""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    def fprop(self, inputs):
        """Forward propagates activations through the layer transformation.

        For inputs `x` and outputs `y` this corresponds to `y = max(0, x)`.
        """
        positive_inputs = np.maximum(inputs, 0.)

        negative_inputs = np.copy(inputs)
        negative_inputs[negative_inputs>0] = 0.
        negative_inputs = self.alpha * (np.exp(negative_inputs) - 1)

        outputs = positive_inputs + negative_inputs
        return outputs

    def bprop(self, inputs, outputs, grads_wrt_outputs):
        """Back propagates gradients through a layer.

        Given gradients with respect to the outputs of the layer calculates the
        gradients with respect to the layer inputs.
        """
        positive_gradients = (outputs >= 0) * grads_wrt_outputs
        outputs_to_use = (outputs < 0) * outputs
        negative_gradients = (outputs_to_use + self.alpha)
        negative_gradients[outputs >= 0] = 0.
        negative_gradients = negative_gradients * grads_wrt_outputs
        gradients = positive_gradients + negative_gradients
        return gradients

    def __repr__(self):
        return 'ELULayer'

class SELULayer(Layer):
    """Layer implementing an element-wise rectified linear transformation."""
    #α01 ≈ 1.6733 and λ01 ≈ 1.0507
    def __init__(self):
        self.alpha = 1.6733
        self.lamda = 1.0507
        self.elu = ELULayer(alpha=self.alpha)
    def fprop(self, inputs):
        """Forward propagates activations through the layer transformation.

        For inputs `x` and outputs `y` this corresponds to `y = max(0, x)`.
        """
        outputs = self.lamda * self.elu.fprop(inputs)
        return outputs

    def bprop(self, inputs, outputs, grads_wrt_outputs):
        """Back propagates gradients through a layer.

        Given gradients with respect to the outputs of the layer calculates the
        gradients with respect to the layer inputs.
        """
        scaled_outputs = outputs / self.lamda
        gradients = self.lamda * self.elu.bprop(inputs=inputs, outputs=scaled_outputs,
                                                grads_wrt_outputs=grads_wrt_outputs)
        return gradients

    def __repr__(self):
        return 'SELULayer'

class TanhLayer(Layer):
    """Layer implementing an element-wise hyperbolic tangent transformation."""

    def fprop(self, inputs):
        """Forward propagates activations through the layer transformation.

        For inputs `x` and outputs `y` this corresponds to `y = tanh(x)`.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).

        Returns:
            outputs: Array of layer outputs of shape (batch_size, output_dim).
        """
        return np.tanh(inputs)

    def bprop(self, inputs, outputs, grads_wrt_outputs):
        """Back propagates gradients through a layer.

        Given gradients with respect to the outputs of the layer calculates the
        gradients with respect to the layer inputs.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).
            outputs: Array of layer outputs calculated in forward pass of
                shape (batch_size, output_dim).
            grads_wrt_outputs: Array of gradients with respect to the layer
                outputs of shape (batch_size, output_dim).

        Returns:
            Array of gradients with respect to the layer inputs of shape
            (batch_size, input_dim).
        """
        return (1. - outputs**2) * grads_wrt_outputs

    def __repr__(self):
        return 'TanhLayer'


class SoftmaxLayer(Layer):
    """Layer implementing a softmax transformation."""

    def fprop(self, inputs):
        """Forward propagates activations through the layer transformation.

        For inputs `x` and outputs `y` this corresponds to

            `y = exp(x) / sum(exp(x))`.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).

        Returns:
            outputs: Array of layer outputs of shape (batch_size, output_dim).
        """
        # subtract max inside exponential to improve numerical stability -
        # when we divide through by sum this term cancels
        exp_inputs = np.exp(inputs - inputs.max(-1)[:, None])
        return exp_inputs / exp_inputs.sum(-1)[:, None]

    def bprop(self, inputs, outputs, grads_wrt_outputs):
        """Back propagates gradients through a layer.

        Given gradients with respect to the outputs of the layer calculates the
        gradients with respect to the layer inputs.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).
            outputs: Array of layer outputs calculated in forward pass of
                shape (batch_size, output_dim).
            grads_wrt_outputs: Array of gradients with respect to the layer
                outputs of shape (batch_size, output_dim).

        Returns:
            Array of gradients with respect to the layer inputs of shape
            (batch_size, input_dim).
        """
        return (outputs * (grads_wrt_outputs -
                           (grads_wrt_outputs * outputs).sum(-1)[:, None]))

    def __repr__(self):
        return 'SoftmaxLayer'


class RadialBasisFunctionLayer(Layer):
    """Layer implementing projection to a grid of radial basis functions."""

    def __init__(self, grid_dim, intervals=[[0., 1.]]):
        """Creates a radial basis function layer object.

        Args:
            grid_dim: Integer specifying how many basis function to use in
                grid across input space per dimension (so total number of
                basis functions will be grid_dim**input_dim)
            intervals: List of intervals (two element lists or tuples)
                specifying extents of axis-aligned region in input-space to
                tile basis functions in grid across. For example for a 2D input
                space spanning [0, 1] x [0, 1] use intervals=[[0, 1], [0, 1]].
        """
        num_basis = grid_dim**len(intervals)
        self.centres = np.array(np.meshgrid(*[
            np.linspace(low, high, grid_dim) for (low, high) in intervals])
        ).reshape((len(intervals), -1))
        self.scales = np.array([
            [(high - low) * 1. / grid_dim] for (low, high) in intervals])

    def fprop(self, inputs):
        """Forward propagates activations through the layer transformation.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).

        Returns:
            outputs: Array of layer outputs of shape (batch_size, output_dim).
        """
        return np.exp(-(inputs[..., None] - self.centres[None, ...])**2 /
                      self.scales**2).reshape((inputs.shape[0], -1))

    def bprop(self, inputs, outputs, grads_wrt_outputs):
        """Back propagates gradients through a layer.

        Given gradients with respect to the outputs of the layer calculates the
        gradients with respect to the layer inputs.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).
            outputs: Array of layer outputs calculated in forward pass of
                shape (batch_size, output_dim).
            grads_wrt_outputs: Array of gradients with respect to the layer
                outputs of shape (batch_size, output_dim).

        Returns:
            Array of gradients with respect to the layer inputs of shape
            (batch_size, input_dim).
        """
        num_basis = self.centres.shape[1]
        return -2 * (
            ((inputs[..., None] - self.centres[None, ...]) / self.scales**2) *
            grads_wrt_outputs.reshape((inputs.shape[0], -1, num_basis))
        ).sum(-1)

    def __repr__(self):
        return 'RadialBasisFunctionLayer(grid_dim={0})'.format(self.grid_dim)

class DropoutLayer(StochasticLayer):
    """Layer which stochastically drops input dimensions in its output."""

    def __init__(self, rng=None, incl_prob=0.5, share_across_batch=True):
        """Construct a new dropout layer.

        Args:
            rng (RandomState): Seeded random number generator.
            incl_prob: Scalar value in (0, 1] specifying the probability of
                each input dimension being included in the output.
            share_across_batch: Whether to use same dropout mask across
                all inputs in a batch or use per input masks.
        """
        super(DropoutLayer, self).__init__(rng)
        assert incl_prob > 0. and incl_prob <= 1.
        self.incl_prob = incl_prob
        self.share_across_batch = share_across_batch
        self.rng = rng

    def fprop(self, inputs, stochastic=True):
        """Forward propagates activations through the layer transformation.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).
            stochastic: Flag allowing different deterministic
                forward-propagation mode in addition to default stochastic
                forward-propagation e.g. for use at test time. If False
                a deterministic forward-propagation transformation
                corresponding to the expected output of the stochastic
                forward-propagation is applied.

        Returns:
            outputs: Array of layer outputs of shape (batch_size, output_dim).
        """
        if stochastic:
            mask_shape = (1,) + inputs.shape[1:] if self.share_across_batch else inputs.shape
            self._mask = (self.rng.uniform(size=mask_shape) < self.incl_prob)
            return inputs * self._mask
        else:
            return inputs * self.incl_prob

    def bprop(self, inputs, outputs, grads_wrt_outputs):
        """Back propagates gradients through a layer.

        Given gradients with respect to the outputs of the layer calculates the
        gradients with respect to the layer inputs. This should correspond to
        default stochastic forward-propagation.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).
            outputs: Array of layer outputs calculated in forward pass of
                shape (batch_size, output_dim).
            grads_wrt_outputs: Array of gradients with respect to the layer
                outputs of shape (batch_size, output_dim).

        Returns:
            Array of gradients with respect to the layer inputs of shape
            (batch_size, input_dim).
        """
        return grads_wrt_outputs * self._mask

    def __repr__(self):
        return 'DropoutLayer(incl_prob={0:.1f})'.format(self.incl_prob)

class MaxPoolingLayer(Layer):

    def __init__(self, pool_size=2, stride=2):
        """Construct a new max-pooling layer.

        Args:
            pool_size: Positive integer specifying size of pools over
               which to take maximum value. The outputs of the layer
               feeding in to this layer must have a dimension which
               is a multiple of this pool size such that the outputs
               can be split in to pools with no dimensions left over.
        """
        self.pool_size = pool_size
        self.stride = stride

    def fprop(self, inputs):

        assert inputs.shape[0] % self.pool_size == 0 and inputs.shape[-1] % self.pool_size == 0, (
            'Last dimension of inputs must be multiple of pool size')

        N, C, H, W = inputs.shape
        out_h = H // self.stride
        out_w = W // self.stride
        col = np.zeros((0,self.pool_size**2))
        for i in inputs:
            for j in i:
                for h in range(0, H, self.stride):
                    for w in range(0, W, self.stride):
                        col = np.append(col,j[h:h+self.pool_size, w:w+self.pool_size].reshape((1,-1)),axis=0)
        outcol = np.max(col, axis=1)
        out = np.zeros((N, C, out_h, out_w))
        i = 0
        for n in range(N):
            for c in range(C):
                for h in range(out_h):
                    for w in range(out_w):
                        out[n, c, h, w] = outcol[i]
                        i+=1

        self.arg_max = np.argmax(col, axis=1)
        return out


    def bprop(self, inputs, outputs, grads_wrt_outputs):
        N, C, H, W = inputs.shape
        out_h = (H - self.pool_size)//self.stride + 1
        out_w = (W - self.pool_size)//self.stride + 1
        dout = grads_wrt_outputs.transpose(0, 2, 3, 1)
        pool_sizes = self.pool_size* self.pool_size
        dmax = np.zeros((dout.size, pool_sizes))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_sizes,))
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dcol = dcol.reshape(N, out_h, out_w, C, self.pool_size, self.pool_size).transpose(0, 3, 4, 5, 1, 2)
        dx = np.zeros((N, C, H + self.stride - 1, W + self.stride - 1))
        for y in range(self.pool_size):
            y_max = y + self.stride*out_h
            for x in range(self.pool_size):
                x_max = x + self.stride*out_w
                dx[:, :, y:y_max:self.stride, x:x_max:self.stride] += dcol[:, :, y, x, :, :]
        return dx[:, :, :H, :W]

    def __repr__(self):
        return 'MaxPoolingLayer(pool_size={0})'.format(self.pool_size)

class ReshapeLayer(Layer):
    """Layer which reshapes dimensions of inputs."""

    def __init__(self, output_shape=None):
        """Create a new reshape layer object.

        Args:
            output_shape: Tuple specifying shape each input in batch should
                be reshaped to in outputs. This **excludes** the batch size
                so the shape of the final output array will be
                    (batch_size, ) + output_shape
                Similarly to numpy.reshape, one shape dimension can be -1. In
                this case, the value is inferred from the size of the input
                array and remaining dimensions. The shape specified must be
                compatible with the input array shape - i.e. the total number
                of values in the array cannot be changed. If set to `None` the
                output shape will be set to
                    (batch_size, -1)
                which will flatten all the inputs to vectors.
        """
        self.output_shape = (-1,) if output_shape is None else output_shape

    def fprop(self, inputs):
        """Forward propagates activations through the layer transformation.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).

        Returns:
            outputs: Array of layer outputs of shape (batch_size, output_dim).
        """
        return inputs.reshape((inputs.shape[0],) + self.output_shape)

    def bprop(self, inputs, outputs, grads_wrt_outputs):
        """Back propagates gradients through a layer.

        Given gradients with respect to the outputs of the layer calculates the
        gradients with respect to the layer inputs.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).
            outputs: Array of layer outputs calculated in forward pass of
                shape (batch_size, output_dim).
            grads_wrt_outputs: Array of gradients with respect to the layer
                outputs of shape (batch_size, output_dim).

        Returns:
            Array of gradients with respect to the layer inputs of shape
            (batch_size, input_dim).
        """
        return grads_wrt_outputs.reshape(inputs.shape)

    def __repr__(self):
        return 'ReshapeLayer(output_shape={0})'.format(self.output_shape)

class PrintLayer(LayerWithParameters):
    def __init__(self):
        self.fwd_passed = False
        self.bwd_passed = False
        self.param_passed = False

    def fprop(self, inputs):
        if not self.fwd_passed:
            print('PrintLayer fprop:', inputs.shape)
            self.fwd_passed = True
        return inputs

    def bprop(self, inputs, outputs, grads_wrt_outputs):
        if not self.bwd_passed:
            print('PrintLayer bprop:', grads_wrt_outputs.shape )
            self.bwd_passed = True
        return grads_wrt_outputs

    def grads_wrt_params(self, inputs, grads_wrt_outputs):
        if not self.param_passed:
            print('PrintLayer param:', grads_wrt_outputs.shape )
            self.param_passed = True
        return [np.zeros((1))]   # late EDIT  !!!

    @property
    def params(self):
        """A list of layer parameter values: `[kernels, biases]`."""
        return [np.zeros((1))]

    @params.setter
    def params(self, values):
        pass

    def __repr__(self):
        return 'PrintLayer'
