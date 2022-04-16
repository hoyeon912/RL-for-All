import gym
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Input, Dense

env = gym.make('CartPole-v1')
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

model = Sequential()
model.add(Input(shape=(input_size, )))
model.add(Dense(output_size, activation='relu'))
model.compile(optimizer='sgd', loss='mse')

dis = 0.99
num_episodes = 2000

rList = []
for i in range(num_episodes):
    state = np.array([env.reset()])
    rAll = 0
    done = False
    step_count = 0

    while not done:
        Q = model.predict(state)
        action = np.argmax(Q)
        new_state, reward, done, info = env.step(action)
        new_state = np.array([new_state])
        if done:
            Q[0][action] = reward
        else:
            Q[0][action] = reward + dis * np.max(model.predict(new_state))
        model.fit(state, Q)
        state = new_state
        rAll += reward
        step_count += 1
        if step_count > 1000:
            break
    rList.append(rAll)

print('Score over time: ' + str(sum(rList)/num_episodes))
print('Final Q-Table')
print('Q')
plt.bar(range(len(rList)), rList, color='blue')
plt.show()