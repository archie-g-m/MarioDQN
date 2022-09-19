import torch
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from DQNTrainer import DQNTrainer


if __name__ == "__main__":
    device = torch.device("mps")
    
    train_env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0") #Gym Environemnt
    train_env = JoypadSpace(train_env, [['NOOP'], ["right"], ["right", "A"]])
    train_env.seed(1)

    trainer = DQNTrainer(config_path="dqn_paremeters.cfg", 
                         env=train_env, 
                         device=device)
    
    trainer.train()