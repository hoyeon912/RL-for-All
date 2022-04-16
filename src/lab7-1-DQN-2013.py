from collections import deque
import random
import time
import gym
import numpy as np
from keras import Sequential
from keras.layers import Input, Dense

class DQN:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
    
    def build(self) -> Sequential:
        self.model = Sequential([
            Input((self.input_size, )),
            Dense(100, activation='relu'),
            Dense(100, activation='relu'),
            Dense(100, activation='relu'),
            Dense(self.output_size)
        ])
        self.model.compile(optimizer='sgd', loss='mse')
    
    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        return self.model.predict(x)
    
    def fit(self, x, y):
        return self.model.fit(x, y, batch_size=10)
    

class ReplayMemory:
    def __init__(self, buffer_size, dis):
        self.buffer_size = buffer_size
        self.buffer = deque()
        self.dis = dis

    def append(self, data):
        self.buffer.append(data)
        if len(self.buffer) > self.buffer_size:
            self.buffer.popleft()

    def replay_train(self, network):
        x_stack = np.empty(0).reshape(0, network.input_size)
        y_stack = np.empty(0).reshape(0, network.output_size)
        for _ in range(50):
            batch = random.sample(self.buffer, 10)
            for state, action, reward, next_state, done in batch:
                q = network.predict(state)
                if done:
                    q[0, action] = reward
                else:
                    q[0, action] = reward + self.dis * np.max(network.predict(next_state))
                y_stack = np.vstack([y_stack, q])
                x_stack = np.vstack([x_stack, state])
        return network.fit(x_stack, y_stack)

class Env:
    def __init__(self, env, num_episodes):
        self.env = gym.make(env)
        self.num_episodes = num_episodes
    
    def get_input_size(self):
        return self.env.observation_space.shape[0]

    def get_output_size(self):
        return self.env.action_space.n

    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)
    
    def sample(self):
        return self.env.action_space.sample()

def bot_play(q, env):
    state = env.reset()
    reward_sum = 0
    while True:
        action = np.argmax(q.predict(state))
        state, reward, done, _ = env.step(action)
        reward_sum += reward
        if done:
            print(f'Total score: {reward_sum}')
            break

def main():
    env = Env('CartPole-v1', 5000)
    rm = ReplayMemory(50000, 0.9)
    q = DQN(env.get_input_size(), env.get_output_size())
    q.build()
    step_count = 0
    start = time.time()
    for episode in range(env.num_episodes):
        e = 1. / ((episode/10) + 1)
        done = False
        state = env.reset()
        while not done:
            if np.random.rand(1) < e:
                action = env.sample()
            else:
                action = np.argmax(q.predict(state))
            next_state, reward, done, _ = env.step(action)
            rm.append((state, action, reward, next_state, done))
            state = next_state
            step_count += 1
            if step_count > 10000:
                break
        if (episode + 1) % 10 == 0:
            end = time.time()
            print(f'Episode: {episode-9}-{episode}\tavg_steps: {step_count/10}\tavg_times: {(end-start)/10}')
            rm.replay_train(q)
            step_count = 0
            start = time.time()
    bot_play(q, env)

if __name__ == '__main__':
    main()