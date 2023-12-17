import numpy as np


class Tensor:
    def __init__(self, shape, history=None):
        self.shape = shape
        self.history = history or []

    def __str__(self):
        return f'Tensor(shape={self.shape})'


def int_to_size(n):
    if isinstance(n, int):
        return n, n
    return n


def summarize_modules(modules):
    lines = []
    name_length = max(max(len(m.desc.get('name', '')) for m in modules) + 3, 7)

    desc_name = 'DESCRIPTION'
    prop_name = 'LAYER'
    output_shape = 'OUTPUT SHAPE'
    num_weights = 'NUM WEIGHTS'
    num_ops = 'NUM OPS'
    lines.append(f'{prop_name:<{name_length}} {desc_name:<30} {output_shape:<20} {num_weights:<13} {num_ops:<10}')
    lines.append('-' * len(lines[0]))
    for module in modules:
        prop_name = str(module.desc.get('name', ''))
        desc = str(module.desc.get('desc', ''))
        output_shape = str(module.desc.get('output_shape', ''))
        num_weights = str(module.desc.get('num_weights', ''))
        num_ops = str(module.desc.get('num_ops', ''))
        lines.append(f'{prop_name:<{name_length}} {desc:<30} {output_shape:<20} {num_weights:<13} {num_ops:<10}')
    return '\n'.join(lines)


class Module:
    def __init__(self):
        pass

    def forward(self, x):
        raise AssertionError('forward of module need to be overwritten.')

    def __call__(self, x):
        return self.forward(x)

    def get_submodules(self):
        submodules = {}
        for prop_name, value in vars(self).items():
            if type(value) in (Conv2d, Linear, MaxPool2d):
                submodules[prop_name] = value
        return submodules

    def summary(self, x):
        output = self(x)
        for name, submodule in self.get_submodules().items():
            submodule.desc['name'] = name
        return summarize_modules(output.history)


class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, padding_mode='zeros'):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = int_to_size(kernel_size)
        self.stride = int_to_size(stride)
        self.padding = int_to_size(padding)
        self.bias = bias

        self.desc = {
            'desc': f'Conv {self.kernel_size[0]}x{self.kernel_size[1]} in={self.in_channels} out={self.out_channels}',
            'name': 'Conv2d'
        }

    def __call__(self, x):
        if not isinstance(x, Tensor):
            raise ValueError('Unknown input type: {}'.format(type(x).__name__))
        if len(x.shape) != 4:
            raise ValueError(
                f'Input for Conv2d is not a feature map of shape (BATCH_SIZE, CHANNELS, WIDTH, HEIGHT): {x.shape}'
            )
        batch_size = x.shape[0]
        in_channels = x.shape[1]
        if in_channels != self.in_channels:
            raise ValueError(f'Got {in_channels} input channels, but this layer only accepts {self.in_channels}.')

        width = (x.shape[2] - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[0] + 1
        height = (x.shape[3] - self.kernel_size[1] + 2 * self.padding[1]) // self.stride[1] + 1
        output_shape = (batch_size, self.out_channels, width, height)

        self.desc['output_shape'] = output_shape
        self.desc['num_weights'] = (np.prod(self.kernel_size) * self.in_channels + int(self.bias)) * self.out_channels
        self.desc['num_ops'] = np.prod(output_shape[1:]) * self.in_channels * np.prod(self.kernel_size)

        return Tensor(output_shape, history=x.history + [self])


class MaxPool2d:
    def __init__(self, kernel_size, stride=None, padding=0):
        self.kernel_size = int_to_size(kernel_size)
        if stride is None:
            stride = kernel_size
        self.stride = int_to_size(stride)
        self.padding = int_to_size(padding)

        self.desc = {
            'desc': f'MaxPool {self.kernel_size[0]}x{self.kernel_size[1]} stride={self.stride}',
            'name': 'MaxPool2d'
        }

    def __call__(self, x):
        if not isinstance(x, Tensor):
            raise ValueError('Unknown input type: {}'.format(type(x).__name__))
        if len(x.shape) != 4:
            raise ValueError(
                'Input for MaxPool2d is not a feature map of shape (BATCH_SIZE, WIDTH, HEIGHT): {}'.format(x.shape)
            )
        batch_size = x.shape[0]
        n_channels = x.shape[1]
        width = (x.shape[2] - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[0] + 1
        height = (x.shape[3] - self.kernel_size[1] + 2 * self.padding[1]) // self.stride[1] + 1
        output_shape = (batch_size, n_channels, width, height)

        self.desc['output_shape'] = output_shape
        self.desc['num_ops'] = np.prod(output_shape[1:]) * np.prod(self.kernel_size)
        return Tensor(output_shape, history=x.history + [self])


class Linear:
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.desc = {
            'desc': f'Linear {self.in_features} -> {self.out_features}',
            'name': 'Linear'
        }

    def __call__(self, x):
        if not isinstance(x, Tensor):
            raise ValueError('Unknown input type: {}'.format(type(x).__name__))
        if len(x.shape) != 2:
            raise ValueError(
                'Input for Linear is not a feature map of shape (BATCH_SIZE, IN_FEATURES): {}'.format(x.shape)
            )
        if self.in_features != x.shape[1]:
            raise ValueError(
                f'Invalid input shape ({x.shape[1]}) for Linear layers with in_features={self.in_features}'
            )

        new_shape = (x.shape[0], self.out_features)
        self.desc['output_shape'] = new_shape
        self.desc['num_weights'] = (self.in_features + int(self.bias)) * self.out_features
        self.desc['num_ops'] = (self.in_features + int(self.bias)) * self.out_features

        return Tensor(new_shape, history=x.history + [self])


class Flatten:
    def __init__(self):
        self.desc = {
            'name': 'Flatten'
        }


class ReLU:
    def __init__(self):
        self.desc = {
            'name': 'ReLU'
        }
