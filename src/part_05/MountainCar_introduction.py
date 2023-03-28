import gymnasium as gym
import numpy as np
import warnings
warnings.simplefilter("ignore")

env = gym.make('MountainCar-v0')
env.reset()

print('State space: ', env.observation_space)
print('Action space: ', env.action_space)

print(env.observation_space.low)
print(env.observation_space.high)

num_states = (env.observation_space.high - env.observation_space.low)*np.array([10, 100])
num_states = np.round(num_states, 0).astype(int) + 1

(state, _) = env.reset()

state_adj = (state - env.observation_space.low)*np.array([10, 100])
state_adj = np.round(state_adj, 0).astype(int)

state, reward, done, truncated, info = env.step(2)
print(f'{state} , {reward}, {done}, {truncated}, {info}')