import gym

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model
import numpy as np
from collections import deque
from sklearn.utils import shuffle
import copy
from dqn_agent import QLearningMethod
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Варианты модели')
parser.add_argument('--mode', choices=['long', 'short'], default='long', help = 'long - для модели с большой памятью, short - для модели с маленькой')
args = parser.parse_args()

mode = args.mode
rewards = []
avg_rewards = []
reward2break = [100, -100, 10, 200]
eps_params = {'max_eps': 1, 'eps': 1, 'min_eps': 0.05, 'eps_delta': 0.096, 'n_jumps': 10}
agent = QLearningMethod(gamma=0.99, learning_rate=0.0002, batch_size=30, max_memory_len = 1500, tau = 0.001,
                        n_actions=4, n_states=8,
                        eps_settings=eps_params, mode='test') #gamma=0.98, learning_rate=0.0001, batch_size=30, n_actions=4, n_states=8
agent.local_model = load_model('./saved_models/big_memory_smart_agent_local.h5') if mode == 'long' else load_model('./saved_models/short_memory_smart_agent_local.h5')
env = gym.make('LunarLander-v2')
for episod in range(3): #количество эпизодов
    episode_reward = 0
    curr_state = env.reset()
    for time_step in range(1000):
        curr_action = agent.choose_action(curr_state)
        # img.set_data(env.render(mode='rgb_array'))
        next_state, curr_reward, done, _ = env.step(curr_action)
        env.render()
        curr_state = next_state
        episode_reward += curr_reward
        if done or curr_reward in reward2break:
            break
    rewards.append(episode_reward)
print('Награды за эпизоды: ', rewards)