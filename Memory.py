import random
import numpy as np
from collections import deque, namedtuple

stateChange = namedtuple(
    'StateChange', ('state', 'action', 'next_state', 'reward'))


class Memory(object):
    def __init__(self, capacity: int):
        """Initializes Memory object to track state changes

        Args:
            capacity (int): number of state changes to keep in memory
        """
        self.memory = deque([], maxlen=capacity)

    def push(self, state: np.ndarray, action: int, next_state: np.ndarray, reward: float):
        """Adds a stateChange to the memory

        Args:
            state (np.ndarray): original state used to determine action
            action (int): action taken at this state
            next_state (np.ndarray): state generated from given action
            reward (float): reward given for action
        """
        self.memory.append(stateChange(state, action, next_state, reward))

    def sample(self, batch_size: int):
        """randomly samples a given number of instances from memory

        Args:
            batch_size (int): number of samples to return 
            
        Returns:
            List(stateChange): sampled stateChanges
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
