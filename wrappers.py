"""
A collection of objects thats can wrap / otherwise modify arbitrary neural
network layers.
"""


from activationLayers import (
    SigmoidLayer,
    ReLU,
    Tanh,
    Swish,
    LeakyReLU,
    Iden,
    Softmax,
)
from cost_functions import compute_mse_cost, compute_huber_cost, compute_stable_bce_cost


def get_activation_layer_function(out_type="sig"):
    if out_type == "sig":
        OutLayer = SigmoidLayer
    elif out_type == "relu":
        OutLayer = ReLU
    elif out_type == "tanh":
        OutLayer = Tanh
    elif out_type == "swish":
        OutLayer = Swish
    elif out_type == "leak":
        OutLayer = LeakyReLU
    elif out_type == "iden":
        OutLayer = Iden
    elif out_type == "softmax":
        OutLayer = Softmax
    else:
        raise ValueError(f"{out_type} is not supported.")

    return OutLayer


# TODO: Make these into Enums
ACTIVATION_MAP = {
    "tanh": Tanh,
    "relu": ReLU,
    "leaky_relu": LeakyReLU,
    "linear": Iden,
    "sigmoid": SigmoidLayer,
    # "softplus": nn.Softplus,
    "softmax": Softmax,
    "swish": Swish,
}

LOSS_TYPES = {
    "mse": compute_mse_cost,
    # "mae": nn.functional.l1_loss,
    "cross_entropy": compute_stable_bce_cost,
    "huber": compute_huber_cost,
}
