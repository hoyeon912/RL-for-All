import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v1') # Default is_slipperyy attribute setting is True
Q = np.zeros([env.observation_space.n, env.action_space.n])

learning_rate = 0.85
dis = 0.99
num_episodes = 2000

rList = []
for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False

    while not done:
        action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i+1))
        new_state, reward, done, info = env.step(action)
        Q[state, action] = (1-learning_rate) * Q[state, action] + learning_rate * (reward + dis * np.max(Q[new_state, :]))
        state = new_state
        rAll += reward
    rList.append(rAll)

print('Score over time: ' + str(sum(rList)/num_episodes))
print('Final Q-Table')
print('Q')
plt.bar(range(len(rList)), rList, color='blue')
plt.show()

# Score Lap
# 1. 0.3725
# 2. 0.511
# 3. 0.4555
# 4. 0.371
# 5. 0.5395