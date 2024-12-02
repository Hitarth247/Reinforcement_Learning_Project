import gym
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

class EnvironmentSetup:
    def __init__(self, env_name="ALE/Breakout-v5", frame_skip=4, frame_stack=4, resize_dim=(84, 84)):
        self.env = gym.make(env_name)
        
        # Apply wrappers for preprocessing
        self.env = gym.wrappers.RecordEpisodeStatistics(self.env)
        self.env = gym.wrappers.ResizeObservation(self.env, resize_dim)
        self.env = gym.wrappers.GrayScaleObservation(self.env)
        self.env = gym.wrappers.FrameStack(self.env, frame_stack)
        self.env = MaxAndSkipEnv(self.env, skip=frame_skip) 

    def get_env(self):
        return self.env
