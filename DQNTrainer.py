import math
import torch
import random
import numpy as np
import configparser
import torchvision.transforms as T

from gym import Env
from DQN import DQN
from Memory import Memory, stateChange
from itertools import count


class DQNTrainer(object):
    """
    Much of this Class is derived from the tutorial found at:
    https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    However many aspects of this class are changed to operate in an Object Oriented format,
    and is used as a base for experimenting with Deep Q Learning beyond the work I helped with at
    https://github.com/archie-g-m/ReinforcmentLearningComparison
    """

    def __init__(self, config_path: str, env: Env, device: torch.device):
        # Parse Config File
        config = configparser.ConfigParser()
        config.read_file(open(config_path))

        self.BATCH_SIZE = int(config.get(
            'Training Hyperparameters', 'BATCH_SIZE'))
        self.GAMMA = float(config.get(
            'Training Hyperparameters', 'GAMMA'))
        self.EPS_START = float(config.get(
            'Training Hyperparameters', 'EPS_START'))
        self.EPS_END = float(config.get('Training Hyperparameters', 'EPS_END'))
        self.EPS_DECAY = float(config.get(
            'Training Hyperparameters', 'EPS_DECAY'))
        self.TARGET_UPDATE = int(config.get(
            'Training Hyperparameters', 'TARGET_UPDATE'))
        self.MAX_STEPS = int(config.get(
            'Training Hyperparameters', 'MAX_STEPS'))

        # Load Gym environment
        self.env = env

        # Establish device to run on
        self.device = device

        # Create Policy and Target DQN networks
        self.n_actions = env.action_space.n
        self.policy_dqn = DQN(self.n_actions).to(self.device)
        self.target_dqn = DQN(self.n_actions).to(self.device)

        # Sync both networks to start
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
        self.target_dqn.eval()

        # Create Optimizer
        self.optimizer = torch.optim.RMSprop(
            params=self.policy_dqn.parameters())

        # Create Memory for StateChanges
        self.memory = Memory(1000)

        # Track number steps
        self.steps_done = 0

        # Last State
        self.last_state = None

    def select_action(self, state):
        """generates an action based upon the given input state

        Args:
            state (_type_): input state for model

        Returns:
            torch.tensor: _description_
        """
        # Randomly sample number to determine random action vs policy action
        sample = random.random()
        # Calculate the threshold for
        eps_threshold = self.EPS_END + \
            (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return torch.argmax(self.policy_dqn(state))
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        """
        Function from: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html to optimize the DQN model, 
        changes include altering code to opperate in the Object Oriented environment and potential changes to Loss Function etc.
        """
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Convertt batch-array of Transitions to Transition of batch-arrays.
        batch = stateChange(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      dtype=torch.bool)
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_dqn
        oh_action_batch = torch.nn.functional.one_hot(
            action_batch, num_classes=self.n_actions)
        state_action_values = torch.argmax(self.policy_dqn(
            state_batch).gather(1, oh_action_batch), dim=1).float()
        state_action_values.requires_grad = True

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, dtype=torch.long)
        non_final_next_state_values = torch.argmax(
            self.target_dqn(non_final_next_states), dim=1)
        next_state_values[non_final_mask] = non_final_next_state_values.to(
            torch.device("cpu"))
        # Compute the expected Q values
        expected_state_action_values = (
            (next_state_values * self.GAMMA) + reward_batch.to(torch.device("cpu"))).float()

        # Compute Huber loss
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(state_action_values.to(self.device),
                         expected_state_action_values.to(self.device))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.policy_dqn.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
    def state_to_tensor(self, state: np.ndarray, size: tuple = None):
        t = torch.tensor(state.transpose(2, 0, 1).copy(), dtype=torch.float, device=self.device).unsqueeze(0)
        if size is not None:
            t = T.Resize((size))(t)
        return t

    def train(self):
        for i in range(300):
            self.train_episode(i)

    def train_episode(self, nEpisode: int):
        # Initialize the environment and state
        state = self.env.reset()
        last_state = self.state_to_tensor(state)
        current_state = last_state
        state = current_state - last_state
        for t in range(self.MAX_STEPS):
            # Select an action
            action = self.select_action(state).squeeze().unsqueeze(0)
            # Perform action
            state, reward, done, info = self.env.step(action.item())
            # Convert action and reward to tensors
            state = self.state_to_tensor(state)
            reward = torch.tensor([reward], device=self.device)

            # Observe new state
            last_state = current_state
            current_state = state
            if not done:
                next_state = current_state - last_state
            else:
                next_state = None

            # Store the transition in memory
            self.memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            self.optimize_model()
            if done:
                break
            # Update the target network, copying all weights and biases in DQN
            if nEpisode % self.TARGET_UPDATE == 0:
                self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

            # self.env.render(mode='human')
            if t % 10 == 0:
                print(f"Episode {str(nEpisode).zfill(4)}/300, Step {str(t).zfill(4)}/{self.MAX_STEPS}, Score: {info['score']}, Action: {action.item()}", end="\r")


if __name__ == "__main__":
    trainer = DQNTrainer("dqn_paremeters.cfg")
