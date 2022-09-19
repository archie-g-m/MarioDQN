import torch
import configparser
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym.wrappers import FrameStack
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from EnvWrappers import *

from DQNTrainer import DQNTrainer

import argparse

if __name__ == "__main__":
    # Parse input Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str,
                        metavar="cfg", help="path to config file")
    args = parser.parse_args()

    # Parse Config File
    config = configparser.ConfigParser()
    config.read_file(open(args.config_file))

    #Get Environemnt variables from config file
    frame_skips = int(config.get("Environment Hyperparameters", "SKIP_FRAMES"))
    grayscale = bool(config.get("Environment Hyperparameters", "GRAYSCALE"))
    resize = int(config.get("Environment Hyperparameters", "ENV_SIZE"))
    frame_stack = int(config.get("Environment Hyperparameters", "FRAME_STACK"))

    # Assign torch device
    device = torch.device("mps")
    # device = torch.device("cpu")
    
    #Construct training environment
    train_env = gym_super_mario_bros.make("SuperMarioBros-1-1-v3")  # Gym Environemnt
    train_env = SkipFrame(train_env, skip=frame_skips)
    if grayscale:
        train_env = GrayScaleObservation(train_env)
    train_env = ResizeObservation(train_env, shape=resize)
    train_env = FrameStack(train_env, num_stack=4)# [["right"], ["right", "A"]])
    train_env = JoypadSpace(train_env, SIMPLE_MOVEMENT)
    train_env.seed(1)

    trainer = DQNTrainer(config_path=args.config_file,
                         env=train_env,
                         device=device)

    trainer.train(render_every=50)
