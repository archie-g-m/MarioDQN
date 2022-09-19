import torch
import torch.nn as nn
from torch import Tensor

class DQN(nn.Module):
    def __init__(self, output_size: int):
        """initialized the Neural Network for Deep Q Learning

        Args:
            output_size (int): output size of the neural network (action space of environment)
        """
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        
        self.fn1 = nn.LazyLinear(128)
        self.fn2 = nn.LazyLinear(output_size)

    def forward(self, input: Tensor) -> Tensor:
        """forward function of neural network

        Args:
            input (Tensor): input array to DQN Neural Network

        Returns:
            Tensor: output of DQN Neural Network
        """
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fn1(x)
        output = self.fn2(x)
        return output