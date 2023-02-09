import numpy as np
import torch
import torch.nn as nn


class BinaryMorphology(nn.Module):
    def __init__(self, kernel_size=5, type=None):
        '''
        n_channels: int
        kernel_size: scalar, the spatial size of the morphological neure.
        type: str, dilate or erode.
        '''
        super().__init__()
        assert type in ['dilate', 'erode'], (f'Invalid `type` {type}')

        self.kernel_size = kernel_size
        self.opp_type = type

        kernel = np.ones((1, 1, kernel_size, kernel_size), dtype=np.float32)
        self.register_buffer('kernel', torch.Tensor(kernel))


    def forward(self, x):
        if self.opp_type == 'dilate':
            x = torch.clamp(torch.nn.functional.conv2d(
                    torch.ge(x, 0.5).float(), self.kernel,
                    padding=self.kernel_size // 2), 0, 1)
        else: # erode
            x = 1 - torch.clamp(torch.nn.functional.conv2d(
                torch.lt(x, 0.5).float(), self.kernel,
                padding=self.kernel_size // 2), 0, 1)

        return x 

class Dilate2D(BinaryMorphology):
    def __init__(self, kernel_size=5):
        super().__init__(kernel_size, 'dilate')

class Erode2D(BinaryMorphology):
    def __init__(self, kernel_size=5):
        super().__init__(kernel_size, 'erode')
