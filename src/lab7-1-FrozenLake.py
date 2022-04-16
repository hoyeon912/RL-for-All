from collections import deque
import random
import gym
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Input, Dense

def one_hot(x):
    return np.identity(16)[x:x+1]

def replay_train(batch):
    x_stack = np.empty(0).reshape(0, input_size)
    y_stack = np.empty(0).reshape(0, output_size)

    for state, action, reward, next_state, done in batch:
        Q = model.predict(state)
        if done:
            Q[0, action] = reward
        else:
            Q[0, action] = reward + dis * np.max(model.predict(next_state))
        x_stack = np.vstack([x_stack, state])
        y_stack = np.vstack([y_stack, 0])
    return model.fit(x_stack, y_stack)

env = gym.make('FrozenLake-v1')
input_size = env.observation_space.shape
output_size = env.action_space.n

model = Sequential()
model.add(Input(shape=input_size))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(output_size, activation='relu'))
model.compile(optimizer='sgd', loss='mse')

dis = 0.9
num_episodes = 2000
buffer = deque()
buffer_size = 50000

rList = []
for episode in range(num_episodes):
    state = one_hot(env.reset())
    rAll = 0
    done = False

    while not done:
        Q = model.predict(state)
        action = np.argmax(Q)
        next_state, reward, done, info = env.step(action)
        next_state = one_hot(next_state)
        buffer.append([state, action, reward, next_state, done])
        if len(buffer) > buffer_size:
            buffer.popleft()
        state = next_state
        rAll += reward
    rList.append(rAll)

    if episode % 10 == 1:
        for _ in range(50):
            batch = random.sample(buffer, 10)
            hist = replay_train(batch)

state = one_hot(env.reset())
num_episodes = 0
while True:
    num_episodes += 1
    env.rendor()
    Q = model.predict(state)
    action = np.argmax(Q)
    next_state, reward, done, info = env.step(action)
    next_state = one_hot((next_state))
    if done and reward:
        break
print(f'Clear the game at {num_episodes}st episode')

print('Score over time: ' + str(sum(rList)/num_episodes))
plt.bar(range(len(rList)), rList, color='blue')
plt.show()