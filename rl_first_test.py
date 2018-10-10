import gym
import gym_sokoban
import numpy as np
import time

import random
from IPython.display import clear_output

def map_np(a):
    return hash(str(a))

env = gym.make("Sokoban-small-v0")
q_table = {}

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# For plotting metrics
all_epochs = []
all_penalties = []

for i in range(0, 1000):
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False
    
    while not done:
        if map_np(state) not in q_table:
            q_table[map_np(state)] = np.zeros(env.action_space.n)
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[map_np(state)]) # Exploit learned values
        

        next_state, reward, done, info = env.step(action)
        
        old_value = q_table[map_np(state)][action]
        next_max = np.max(q_table[map_np(next_state)])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[map_np(state)][action] = new_value

        # env.render("human")
        # time.sleep(.01)

        if reward < 0:
            penalties += 1

        state = next_state

        epochs += 1
        
    print("Episode: " + str(i))

print(q_table)

print("Training finished.\n")

total_epochs, total_penalties = 0, 0
episodes = 10

for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    
    done = False
    
    while not done:
        action = np.argmax(q_table[state])

        state, reward, done, info = env.step(action)

        if reward < 0:
            penalties += 1

        epochs += 1
        env.render(mode="human")

    total_penalties += penalties
    total_epochs += epochs

print("Results after " + str(episodes) + " episodes:")
print("Average timesteps per episode: " + str(total_epochs / episodes))
print("Average penalties per episode: " + str(total_penalties / episodes))
env.close()