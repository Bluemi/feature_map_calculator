import numpy as np
from nn import Tensor, ReLU, Flatten


def relu(x):
    if not isinstance(x, Tensor):
        raise ValueError('Unknown input type: {}'.format(type(x).__name__))
    tmp_layer = ReLU()
    tmp_layer.desc['output_shape'] = x.shape
    tmp_layer.desc['name'] = 'ReLU'
    return Tensor(x.shape, history=x.history + [tmp_layer.desc])


def flatten(x, start_dim=0):
    if not isinstance(x, Tensor):
        raise ValueError('Unknown input type: {}'.format(type(x).__name__))

    output_shape = (*x.shape[:start_dim], np.prod(x.shape[start_dim:]))

    tmp_layer = Flatten()
    tmp_layer.desc['output_shape'] = output_shape
    tmp_layer.desc['name'] = 'Flatten'
    return Tensor(output_shape, history=x.history + [tmp_layer.desc])
