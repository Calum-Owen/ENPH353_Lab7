#!/usr/bin/env python3
import random
from pathlib import Path
import gym_gazebo
import gymnasium as gym
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter

def main():
    writer = SummaryWriter()
    env = gym.make('LineFollowerEnv-v0')

    observation_space = env.observation_space
    action_space = env.action_space

    epsilon = 0.00001
    epsilon_decay = 0.999

    q_path = Path(__file__).resolve().with_name("linefollower_q.npy")
    expected_shape = (observation_space.n, action_space.n)
    if q_path.exists():
        q_table = loadQ(q_path)
        if q_table.shape != expected_shape:
            raise ValueError(
                f"Saved Q-table has shape {q_table.shape}; expected {expected_shape}"
            )
        print(f"Loaded Q-table from {q_path}")
    else:
        q_table = np.zeros(expected_shape)
        print("No saved Q-table found; starting with zeros.")

    highest_reward = 0

    try:
        state, _ = env.reset()
        for episode in range(10000):
            total_reward = 0
            actions_per_episode = 0
            done = False

            while not done:
                actions_per_episode += 1
                action = choose_action(
                    state, q_table, epsilon, action_space, observation_space.start
                )
                if actions_per_episode <= 5:
                    action = 0
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                done = terminated or truncated

                q_table = learn(
                    state,
                    action,
                    next_state,
                    reward,
                    q_table,
                    terminated,
                    observation_space.start,
                )

                state = next_state
                total_reward += reward

            # Log data
            print(f"Episode {episode}: Epsilon: {epsilon} | Total Reward: {total_reward}")
            print("Q values:")
            print(q_table)
            writer.add_scalar("Total Reward", total_reward, episode)
            writer.add_scalar("Actions per episode", actions_per_episode, episode)
            saveQ(q_table, q_path)

            if total_reward > highest_reward:
                highest_reward = total_reward
                print("New highest reward: {total_reward}.")

            state, _ = env.reset()
            epsilon = max(0.01, epsilon * epsilon_decay)  # Decay epsilon

    except KeyboardInterrupt:
        print("Training interrupted by user!")
    
    finally:
        saveQ(q_table, q_path)
        env.close()
        writer.flush()
        writer.close()

def learn(state, action, next_state, reward, q_table, terminated, state_start=0):
    alpha = 0.01
    gamma = 0.9

    Bellman_Error = reward
    if not terminated:
        next_state_index = next_state - state_start
        Bellman_Error += gamma * np.max(q_table[next_state_index])

    state_index = state - state_start
    q_table[state_index, action] += alpha * (
        Bellman_Error - q_table[state_index, action]
    )

    return q_table
    

#TODO: action_space filter if this code is reused in later lab 
def choose_action(state, q_table, epsilon, action_space, state_start=0):


    if random.random() < epsilon:
        return action_space.sample()
    
    state_index = state - state_start
    q_values = q_table[state_index]
    best_actions = np.flatnonzero(q_values == np.max(q_values))
    return int(random.choice(best_actions))



def saveQ(q_table, save_path):
    q_table = np.asarray(q_table)
    if q_table.ndim != 2:
        raise ValueError("Q table must be a 2D array")

    with open(save_path, "wb") as q_file:
        np.save(q_file, q_table, allow_pickle=False)

def loadQ(load_path):
    q_table = np.load(load_path, allow_pickle=False)
    if q_table.ndim != 2:
        raise ValueError("Saved Q table must be a 2D array")
    return q_table

if __name__ == "__main__":
    main()