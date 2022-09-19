import torch
import torch.nn as nn
from torch import Tensor

class DQN(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        """initialized the Neural Network for Deep Q Learning

        Args:
            output_size (int): output size of the neural network (action space of environment)
        """
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 32, kernel_size=3, stride=1)
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
        # print(f"Input: {input.shape}")
        x = self.conv1(input)
        # print(f"Conv1: {x.shape}")
        x = self.conv2(x)
        # print(f"Conv2: {x.shape}")
        x = self.conv3(x)
        # print(f"Conv3: {x.shape}")
        x = self.flatten(x)
        # print(f"Flatten: {x.shape}")
        x = self.fn1(x)
        # print(f"FN1: {x.shape}")
        output = self.fn2(x)
        # print(f"OUT: {output.shape}")
        return output