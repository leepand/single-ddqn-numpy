import numpy as np  # import numpy library
from typing import Union, Dict
from functional import Activation

from utils import (
    initialize_parameters,
)  # import function to initialize weights and biases


class Input(object):
    """
    Input layer
    """

    def __init__(self, input_dim: int) -> None:
        """
        Store the input dimension.

        Parameters:
        - input_dim (int): The dimension of the input.

        Returns:
        - None
        """
        self.n_out = input_dim


class Dense:
    """
    Dense layer
    This Class implements all functions to be executed by a linear layer
    in a computational graph

    Args:
        input_shape: input shape of Data/Activations
        n_out: number of neurons in layer
        ini_type: initialization type for weight parameters, default is "plain"
                  Opitons are: plain, xavier and he

    Methods:
        forward(A_prev)
        backward(upstream_grad)
        update_params(learning_rate)

    """

    def __init__(self, units, n_out=None, activation="tanh", activation_params={}):
        """
        The constructor of the LinearLayer takes the following parameters

        Args:
            input_shape: input shape of Data/Activations
            n_out: number of neurons in layer
            ini_type: initialization type for weight parameters, default is "plain"
        """
        self.layer_type = "Linear"
        self.activation_name = activation
        self.activation_params = activation_params
        self.n_in = units
        self.n_out = n_out
        self.m = 1  # number of examples in training data
        # `params` store weights and bias in a python dictionary
        # momentum
        # self.m1 = np.random.uniform(0.1, 1, self.w.shape)
        # self.m2 = np.random.uniform(0.1, 1, self.w.shape)

    def get_activation_name(self) -> str:
        return self.activation_name

    def forward(self, A_prev, W, b, predict=True):
        """
        This function performs the forwards propagation using activations from previous layer

        Args:
            A_prev:  Activations/Input Data coming into the layer from previous layer
        """

        # self.A_prev = A_prev  # store the Activations/Training Data coming in
        Z = np.dot(W, A_prev) + b  # compute the linear function
        if predict:
            return Z
        else:
            return A_prev, Z

    def backward(self, upstream_grad, A_prev, W):
        """
        This function performs the back propagation using upstream gradients

        Args:
            upstream_grad: gradient coming in from the upper layer to couple with local gradient
        """
        # derivative of Cost w.r.t W
        dW = np.dot(upstream_grad, A_prev.T)
        # derivative of Cost w.r.t b, sum across rows
        db = np.sum(upstream_grad, axis=1, keepdims=True)
        # derivative of Cost w.r.t A_prev
        dA_prev = np.dot(W.T, upstream_grad)
        return dA_prev, dW, db


class Activ(object):
    """
    Activation layer
    """

    def __init__(
        self, activation: str = "linear", activation_params: Dict = {}
    ) -> None:
        """
        Initialize the Activation layer.

        Parameters:
        - activation (str): The name of the activation function.
        - activation_params (dict): The parameters of the activation function.
        """
        self.n_in = None
        self.activation_prev = None
        self.activation_name = activation
        self.activation_obj = Activation(activation, activation_params)
        self.activation_f = self.activation_obj.f
        self.activation_deriv = None
        self.params = {}

    def get_n_in(self, n_in: int) -> None:
        """
        Set the number of input neurons.

        Parameters:
        - n_in (int): The number of input neurons.
        """
        self.n_in = n_in
        self.n_out = n_in
        self.params["W"] = np.ones((n_in, n_in))

    def get_activation_name(self) -> str:
        return self.activation_name

    def add_activation_deriv(self, activation_prev: str) -> None:
        self.activation_deriv = Activation(activation_prev).f_deriv

    def forward(self, x):
        """
        Forward propagation.

        Parameters:
        - x (np.ndarray): The inputs of the layer.
        """
        outputs = self.activation_f(x)
        # self.inputs = x
        return outputs

    def backward(self, dy, x, is_last_layer: bool = False):
        """
        Backward propagation.

        Parameters:
        - delta (np.ndarray): The delta of the layer.
        - is_last_layer (bool): Whether the layer is the last layer.

        Returns:
        - delta (np.ndarray): The delta of the previous layer.
        """
        # Compute the gradients of weights and biases
        # If the layer is not the last layer, compute the delta of the previous layer
        delta = dy
        if not is_last_layer and self.activation_deriv is not None:
            delta *= self.activation_deriv(x)
        return delta
