import gym
import os
import torch
import random
from gymnasium import envs
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.buffers import ReplayBuffer
from agent import DQN
from env import EnvironmentSetup
from types import SimpleNamespace

params = SimpleNamespace(
    replay_memory_size=1000000,
    nb_epochs=3000000,
    update_frequency=4,
    batch_size=64,
    discount_factor=0.99,
    replay_start_size=80000,
    initial_exploration=1.0,
    final_exploration=0.01,
    exploration_steps=1000000,
    device="cpu",
    learning_rate=1.25e-4,
)

def Deep_Q_Learning(env, q_net, rb, optimizer):

    epoch = 0
    smoothed_rewards = []
    rewards = []

    while epoch <= params.nb_epochs:

        dead = False
        total_rewards = 0

        obs = env.reset()

        for _ in range(random.randint(1, 30)):
            obs, _, _, info = env.step(1)

        while not dead:
            info_param = 'lives'
            current_life = info[info_param]

            epsilon = max((params.final_exploration - params.initial_exploration) / params.exploration_steps * epoch + params.initial_exploration,
                          params.final_exploration)
            
            if random.random() < epsilon:
                action = np.array(env.action_space.sample())
            else:
                q_values = q_net(torch.Tensor(obs).unsqueeze(0).to(params.device))
                action = torch.argmax(q_values, dim=1).item()

            next_obs, reward, dead, info = env.step(action)

            done = True if (info['lives'] < current_life) else False

            real_next_obs = next_obs.copy()

            total_rewards += reward
            reward = np.sign(reward)

            rb.add(obs, real_next_obs, action, reward, done, info)

            obs = next_obs

            if epoch > params.replay_start_size and epoch % params.update_frequency == 0:
                data = rb.sample(params.batch_size)
                with torch.no_grad():
                    max_q_value, _ = q_net(data.next_observations).max(dim=1)
                    y = data.rewards.flatten() + params.discount_factor * max_q_value * (1 - data.dones.flatten())
                current_q_value = q_net(data.observations).gather(1, data.actions).squeeze()
                loss = F.huber_loss(y, current_q_value)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch += 1
            if (epoch % 1000 == 0):
                smoothed_rewards.append(np.mean(rewards))
                rewards = []
                plt.plot(smoothed_rewards)
                plt.title("Breakout")
                plt.xlabel("Epochs")
                plt.ylabel("Rewards")
                if not os.path.exists('Imgs'):
                    os.makedirs('Imgs')

                plt.savefig('Imgs/breakout.png')
                plt.close()

        rewards.append(total_rewards)


if __name__ == "__main__":
    env = EnvironmentSetup().get_env()
    rb = ReplayBuffer(params.replay_memory_size, env.observation_space, env.action_space, params.device, optimize_memory_usage=True, handle_timeout_termination=False)
    q_network = DQN(env.action_space.n).to(params.device)
    optimizer = torch.optim.Adam(q_network.parameters(), lr=params.learning_rate)
    Deep_Q_Learning(env, q_network, rb, optimizer)
    env.close()