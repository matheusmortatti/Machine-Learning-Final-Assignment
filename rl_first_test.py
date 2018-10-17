import gym
import gym_sokoban
import numpy as np
import time
import pickle
from hashlib import sha1

import random
from IPython.display import clear_output

def map_np(a):
    return hash(np.ndarray.tostring(a))

env = gym.make("PushAndPull-Sokoban-v2")
q_table = {}

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# For plotting metrics
all_epochs = []
all_penalties = []

try:
    q_table = pickle.load(open('q_table.p', 'rb'))
except FileNotFoundError:
    pass

episodes = 100000

for i in range(episodes):
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False

    #print(len(state))
    #print(len(state[0]))
    #print(len(state[0][0]))
    #print(state)

    while not done:
        mapped_state = map_np(state)
        if mapped_state not in q_table:
            q_table[mapped_state] = np.zeros(env.action_space.n)
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[mapped_state]) # Exploit learned values


        next_state, reward, done, info = env.step(action)
        mapped_next_state = map_np(next_state)

        old_value = q_table[mapped_state][action]

        if mapped_next_state not in q_table:
            q_table[mapped_next_state] = np.zeros(env.action_space.n)
        next_max = np.max(q_table[mapped_next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[mapped_state][action] = new_value

        env.render("human")
        # time.sleep(.01)

        if reward < 0:
            penalties += 1

        # tt = (state == next_state)
        # res = True

        # for i in range(tt.shape[0]):
        #     for j in range(tt.shape[1]):
        #         for k in range(tt.shape[2]):
        #             res = res and tt[i,j,k]
        # print("res: " + str(res))
        # print(str(state.tolist())==str(next_state.tolist))

        state = next_state

        epochs += 1

    print("Episode: " + str(i))

pickle.dump(q_table, open('q_table.p', 'wb'))

print("Training finished.\n")

total_epochs, total_penalties = 0, 0

episodes = 100
for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0

    done = False

    while not done:
        mapped_state = map_np(state)
        if mapped_state not in q_table:
            action = env.action_space.sample()
            #q_table[mapped_state] = np.zeros(env.action_space.n)
        else:
            action = np.argmax(q_table[mapped_state])

        next_state, reward, done, info = env.step(action)

        if reward < 0:
            penalties += 1

        epochs += 1
        env.render(mode="human")

        # tt = (state == next_state)
        # res = True

        # for i in range(tt.shape[0]):
        #     for j in range(tt.shape[1]):
        #         for k in range(tt.shape[2]):
        #             res = res and tt[i,j,k]
        # print(res)

        state = next_state

    total_penalties += penalties
    total_epochs += epochs

print("Results after " + str(episodes) + " episodes:")
print("Average timesteps per episode: " + str(total_epochs / episodes))
print("Average penalties per episode: " + str(total_penalties / episodes))
env.close()
