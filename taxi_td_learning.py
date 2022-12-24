import gym
import gym.spaces
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from collections import deque
from collections import defaultdict
# create the cart-pole environment
env = gym.make('Taxi-v2')


class Taxi():
    def __init__(self, action_size=env.observation_space.n,n_episodes=1000, solved_t=200,
                 min_epsilon=1, min_alpha=1, gamma=0.999):
        self.buckets = action_size  # discrete values for each feature space dimension
        # (position, velocity, angle, angular velocity)
        self.n_episodes = n_episodes  # training episodes
        self.min_alpha = min_alpha
        self.min_epsilon = min_epsilon
        self.gamma = gamma  # discount factor
        self.solved_t = solved_t  # lower bound before episode ends
        self.epsilon = min_epsilon
        self.Q_table = np.zeros([env.observation_space.n, env.action_space.n])



    def select_action(self, state, epsilon):
        # implement the epsilon-greedy approach
        if random.random() <= epsilon:
            return env.action_space.sample()  # sample a random action with probability epsilon
        else:
            return np.argmax(self.Q_table[state])  # choose greedy action with hightest Q-value

    def get_epsilon(self, episode_number,state):
        # choose decaying epsilon in range [min_epsilon, 1]
        if random.random() > self.epsilon:  # select greedy action with probability epsilon
            return np.argmax(self.Q_table[state])
        else:  # otherwise, select an action randomly
            return random.choice(np.arange(env.action_space.n))

    def get_alpha(self, episode_number):
        # choose decaying alpha in range [min_alpha, 1]
        return max(self.min_alpha, min(1, 1 - math.log10((episode_number + 1) / 25)))

    def update_table(self, old_state, action, reward, new_state, alpha,next_state=None, next_action=None):
        # updates the state-action pairs based on future reward
        current = self.Q_table[(new_state)][action]  # estimate in Q-table (for current state, action pair)
        # get value of state, action pair at next time step
        Qsa_next = self.Q_table[(next_state)][next_action] if next_state is not None else 0
        target = reward + (self.gamma * Qsa_next)  # construct TD target
        new_value = current + (alpha * (target - current))  # get updated value
        return new_value

    def run(self):
        # runs episodes until mean reward of last 100 consecutive episodes is atleast self.solved_t
        env.seed(0)
        np.random.seed(0)
        total_epochs, total_penalties = 0, 0
        #counter = 0
        scores = deque(maxlen=200)
        episodes_result = deque(maxlen=200)
        penalties_result = deque(maxlen=100)
        results = []
        for episode in range(self.n_episodes):
            obs = env.reset()
            curr_state = obs
            done = False
            alpha = self.get_alpha(episode)
            epsilon = self.get_epsilon(episode,curr_state)
            epochs,penalties ,episode_reward = 0,0,0

            #results.append(cartpole.run())


            while not done:
                #env.render()
                if random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample()  # Explore action space
                else:
                    action = self.select_action(curr_state, epsilon)
                obs, reward, done, info = env.step(action)
                new_state = obs

                self.update_table(curr_state, action, reward, new_state, alpha)
                curr_state = new_state
                episode_reward += reward
                print('Reward:', reward)
                if reward > -1:
                    penalties += 1
                    print('penalties:', penalties)
                epochs += 1
                total_penalties += penalties
                total_epochs += epochs
            scores.append(episode_reward)
            episodes_result.append(epochs)
            penalties_result.append(total_penalties)
            mean_reward = np.mean(scores)

        if mean_reward > self.solved_t and (episode + 1) >= 100:
            print("Ran {} episodes, solved after {} trials".format(episode + 1, episode + 1 - 100))
            return episode + 1 - 100
        elif (episode + 1) % 50 == 0 and (episode + 1) >= 100:
            print("Episode number: {}, mean reward over past 100 episodes is {}".format(episode + 1, mean_reward))
        else:
            print("Episode {}, reward {}".format(episode + 1, episode_reward))

        print(f"Results after {episode} episodes:")
        print(f"Average timesteps per episode: {total_epochs / episode}")
        print(f"Average Rewards per episode: {np.mean(scores)}")
        print(f"Average penalties per episode: {total_penalties / episode}")
        print("Training finished.\n")
        plt.hist(episodes_result, 50, normed=1, facecolor='g', alpha=0.75)
        plt.xlabel('Episodes required to reach Goal')
        plt.ylabel('Frequency')
        plt.title('Episode Histogram of Taxi problem solving by TD Learning')
        plt.show()
        plt.hist(scores, 50, normed=1, facecolor='g', alpha=0.75)
        plt.xlabel('Rewards Achieved Per Episode')
        plt.ylabel('Frequency')
        plt.title('Rewards Histogram of Taxi problem solving by TD Learning')
        plt.show()
        plt.hist(penalties_result, 50, normed=1, facecolor='g', alpha=0.75)
        plt.xlabel('Penalties Per Episode')
        plt.ylabel('Frequency')
        plt.title('Penalties Histogram of Taxi problem solving by TD Learning')
        plt.show()
        return episodes_result, scores, penalties_result

if __name__ == "__main__":

    Taxi = Taxi()
    Taxi.run()
    results = []
    #results.append(cartpole.run())

    #plt.hist(results, 50, normed=1, facecolor='g', alpha=0.75)
    #plt.xlabel('Episodes required to reach 200')
    #plt.ylabel('Frequency')
    #plt.title('Histogram of Random Search')
    #plt.show()

    #print(np.sum(results) / 1000.0)
