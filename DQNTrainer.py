from selectors import BaseSelector
import sys
import math
import time
from typing import MutableSequence
import torch
import random
import numpy as np
import configparser
import torchvision.transforms as T
import matplotlib.pyplot as plt

from gym import Env
from DQN import DQN
from Memory import Memory, stateChange
from collections import deque
from itertools import count
from PIL import Image


class DQNTrainer(object):
    """
    Much of this Class is derived from the tutorial found at:
    https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    However many aspects of this class are changed to operate in an Object Oriented format,
    and is used as a base for experimenting with Deep Q Learning beyond the work I helped with at
    https://github.com/archie-g-m/ReinforcmentLearningComparison
    """

    def __init__(self, config_path: str, env: Env, device: torch.device, model_in: str = None, model_out: str = "models/best_mario.pt"):
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
        self.X_TIMEOUT = int(config.get(
            'Training Hyperparameters', 'X_TIMEOUT'))
        self.X_THRESH = int(config.get(
            'Training Hyperparameters', 'X_TIMEOUT_THRESH'))
        self.STATE_SIZE = int(config.get(
            'Training Hyperparameters', 'STATE_SIZE'))
        self.FRAME_BUFFER = int(config.get(
            'Environment Hyperparameters', 'FRAME_STACK'))

        print(self.X_TIMEOUT)
        # Load Gym environment
        self.env = env

        # Establish device to run on
        self.device = device

        # Establish export path for best model
        self.model_out = model_out

        # Create Policy and Target DQN networks
        self.n_actions = env.action_space.n
        self.policy_dqn = DQN(
            self.FRAME_BUFFER, self.n_actions).to(self.device)
        self.target_dqn = DQN(
            self.FRAME_BUFFER, self.n_actions).to(self.device)

        # Load model state dict if provided
        if model_in is not None:
            self.start_episode, self.best_reward, start_state_dict = torch.load(model_in)
            self.policy_dqn.load_state_dict(start_state_dict)
            print("Loading Checkpoint")
            print("Start Episode " + str(self.start_episode))
            print("Best Reward " + str(self.best_reward))
        else: # Otherwise initialize the tracking metrics to 0
            self.start_episode = 0
            self.best_reward = 0

        # Sync both networks to start
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
        self.target_dqn.eval()

        # Create Optimizer
        self.optimizer = torch.optim.RMSprop(
            params=self.policy_dqn.parameters())

        # Create Memory for StateChanges
        self.memory = Memory(10000)

        # # Create Buffer for Number of frames to feed to robot
        # self.frame_buffer = deque([], maxlen=self.FRAME_BUFFER)

        # Track number steps
        self.steps_done = 0

        # Last State
        self.last_state = None

        # Tracker for iteration times
        self.it_times = deque([], maxlen=1000)
        self.model_times = deque([], maxlen=1000)
        self.step_times = deque([], maxlen=1000)
        self.opt_times = deque([], maxlen=1000)

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
        eps_threshold = self.calc_rand_prob()
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return torch.argmax(self.policy_dqn(state))
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def calc_rand_prob(self):
        """calculate probability that mario should act randomly

        Returns:
            float: probability
        """
        return self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)

    def avg_splits(self, tracker: MutableSequence):
        """Averages the time splits from a tracker

        Args:
            tracker (MutableSequence): _description_

        Returns:
            float: average time
        """
        if len(tracker) > 0:
            return round(sum(tracker)/len(tracker), 4)
        else:
            return 0.

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
        # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
        #                               dtype=torch.bool)
        # non_final_next_states = torch.cat(
        #     [s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_dqn
        state_action_values = self.policy_dqn(state_batch.to(self.device)).gather(1, action_batch.unsqueeze(0)).squeeze()

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = self.target_dqn(next_state_batch.to(self.device)).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(state_action_values,
                         expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return

    def frame_to_tensor(self, frame: np.ndarray, size: tuple = None):
        t = torch.from_numpy(np.asarray(frame)).unsqueeze(0)
        if t.dtype != torch.float:
            t = T.ConvertImageDtype(torch.float)(t)
        return t

    def train(self, render_every: int = None):
        for i in count(start=self.start_episode):
            if render_every is None:
                self.train_episode(i, False)
            elif i % render_every == 0:
                self.train_episode(i, True)
            else:
                self.train_episode(i, False)


    def train_episode(self, nEpisode: int, render: bool):
        episode_rewards = 0
        # Initialize the environment and state
        state = self.env.reset()
        state = self.frame_to_tensor(state)

        # Initialize X Position Tracking
        last_x = 0
        x_pos_timeout = 0

        it_start = time.time()
        for t in range(self.MAX_STEPS):
            self.it_times.append((time.time()-it_start))
            it_start = time.time()

            # Select an action
            action = self.select_action(state.to(self.device)).squeeze().unsqueeze(0)
            model_split = time.time()
            self.model_times.append(model_split-it_start)

            # Perform action
            next_state, reward, done, info = self.env.step(action.item())
            self.step_times.append(time.time()-model_split)

            # Convert this frame and to tensor and append to frame buffer
            next_state = self.frame_to_tensor(next_state)
            # self.frame_buffer.append(this_frame)

            # Convert reward to tensor
            reward = torch.tensor([reward])#, device=self.device)

            # Total the Rewards for the episode
            episode_rewards += reward

            # break if the game is done
            if done:
                break

            # Store the transition in memory if next_state is not none
            self.memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            opt_start = time.time()
            self.optimize_model()
            opt_split = time.time()
            self.opt_times.append(opt_split-opt_start)

            # Track X-Position to determine whether mario is stuck or not
            if info['x_pos'] - last_x <= self.X_THRESH:
                x_pos_timeout += 1
            else:
                x_pos_timeout = 0

            # If Mario hasnt moved for long enough end episode
            if x_pos_timeout > self.X_TIMEOUT:
                break

            # Update the target network, copying all weights and biases in DQN
            if nEpisode % self.TARGET_UPDATE == 0:
                self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

            # Print status of learning every 60 frames
            if t % 60 == 0:
                str_dict = {
                    "episode_str": f"Episode {nEpisode:03.0f}",
                    "step_str": f"Step {t:04.0f}/{self.MAX_STEPS}",
                    "reward_str": f"Rewards: {episode_rewards.item():05.0f}",
                    "best_reward_str": f"Best Reward: {self.best_reward.item():05.0f}",
                    "score_str": f"Score: {info['score']:05.0f}",
                    "xpos_str": f"X Pos: {info['x_pos']:04.0f}",
                    "time_str": f"Timeout: {x_pos_timeout:03.0f}",
                    "prob_str": f"Rand Prob: {round(self.calc_rand_prob(), 3):05.3f}",
                    "ittime_str": f"It Time: {self.avg_splits(self.it_times):06.4f}",
                    "mem_str": f"Mem: {sys.getsizeof(self.memory.memory)/10**6:06.3f} MB",
                    "padding": "{:15}".format(" ")
                }
                print(", ".join(str_dict.values()), end="\r")

            # render the frame if specified
            if render:
                self.env.render(mode='human')

            last_x = info['x_pos']
        if episode_rewards.item() > self.best_reward.item():
            torch.save((nEpisode, episode_rewards, self.policy_dqn.state_dict()), self.model_out)
            self.best_reward = episode_rewards

        print()


if __name__ == "__main__":
    trainer = DQNTrainer("dqn_paremeters.cfg")
