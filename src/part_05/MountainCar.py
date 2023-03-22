import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from QLearningBox import QLearningBox

#
# Reference: https://gymnasium.farama.org/environments/classic_control/mountain_car/

env = gym.make('MountainCar-v0')

print('State space: ', env.observation_space)
print('Action space: ', env.action_space)

print(env.observation_space.low)
print(env.observation_space.high)

qlearn = QLearningBox(env, 0.1, 0.1, 0.8, 0, 0.99, 100)
qtable = qlearn.train('data/q-table-mountain-car.csv', 'results/rewards_MountainCar-v0')

env = gym.make('MountainCar-v0', render_mode='human')
(state,_) = env.reset()
done = False

while not done:
    env.render()
    state_adj = qlearn.transform_state(state)
    action = np.argmax(qtable[state_adj[0], state_adj[1]])
    state2, reward, done, truncated, _ = env.step(action)
    state = state2

input("enter a key...")
env.close()