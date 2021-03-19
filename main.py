import gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque


EPISODES = 10_000
BATCH_SIZE = 64

LR = 0.01
GAMMA = 0.999

MEMORY_SIZE = 50_000
START_TRAIN_SIZE = 500
UPDATE_FREQ = 10

START_EPSILON = 0.99
END_EPSILON = 0
DECAY_STEPS = 700

SHOW_EVERY = 100


class MyDearAgent():
    def __init__(self):
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.device = torch.device("cpu" if torch.cuda.is_available() else "cpu") # WITHOUT CUDA

        self.env = gym.make('CartPole-v1')

        self.policy_model = self.get_model().to(self.device)
        self.target_model = self.get_model().to(self.device)
        self.target_model.load_state_dict(self.policy_model.state_dict())

        self.optimizer = optim.RMSprop(self.target_model.parameters(), LR)
        self.criterion = nn.SmoothL1Loss()

        self.epsilon = START_EPSILON
        self.epsilon_decay = (END_EPSILON - START_EPSILON) / DECAY_STEPS

        # DISABLE GRADS
        self.policy_model.eval()

    def get_model(self):
        return nn.Sequential(
                nn.Linear(4, 16),
                nn.BatchNorm1d(16),
                nn.SiLU(),
                nn.Linear(16, 32),
                nn.BatchNorm1d(32),
                nn.SiLU(),
                nn.Linear(32, 2)
            )

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def reshape_state(self, state):
        return torch.from_numpy(np.asarray(state, dtype=np.float32).reshape(1, 4)).to(self.device)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.policy_model(state).cpu().detach().numpy())

    def train(self):
        if len(self.memory) < START_TRAIN_SIZE:
            return 

        x_train, y_train = list(), list()
        minibatch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in minibatch:
            current_qs = self.policy_model(state)
            if done:
                new_q = reward
            else:
                future_qs = self.policy_model(next_state)
                new_q = reward + GAMMA * np.max(future_qs[0].cpu().detach().numpy())
            current_qs[0][action] = new_q
            x_train.append(state)
            y_train.append(current_qs)

        x_train = torch.cat(x_train, 0).to(self.device)
        y_train = torch.cat(y_train, 0).to(self.device)

        ### FIT

        self.optimizer.zero_grad()

        outputs = self.target_model(x_train)
        loss = self.criterion(outputs, y_train)
        loss.backward()
        self.optimizer.step()

    def run(self):
        scores = deque(maxlen=100)
        #scores.append(0)

        for ep in range(EPISODES):
            current_state = self.reshape_state(self.env.reset())
            done = False
            score = 0

            if not ep % SHOW_EVERY:
                render = True
                #print(f'Episode - {ep}, mean score - {np.mean(scores)}, epsilon - {self.epsilon}')
            else:
                render = False

            while not done:
                if render:
                    self.env.render()
                action = self.choose_action(current_state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.reshape_state(next_state)
                self.memorize(current_state, action, reward, next_state, done)
                current_state = next_state
                score += 1

            if self.epsilon > END_EPSILON:
                self.epsilon += self.epsilon_decay

            scores.append(score)

            if np.mean(scores) > 195.:
                print(f"SOLVED AFTER {ep} EPISODES")
                break

            self.train()

            if not ep % UPDATE_FREQ:
                self.policy_model.load_state_dict(self.target_model.state_dict())






#MyDearAgent()
#print(MyDearAgent().choose_action([[1, 2, 3, 4]]))
#env = gym.make('CartPole-v1')
#print(env.reset())
#print(env.action_space.n)
#print(env.observation_space.high)
#print(env.observation_space.low)
#for _ in range(1000):
#    env.render()
#    a = env.step(env.action_space.sample())
#    print(a)
#    break
#env.close()


if __name__ == '__main__':
    agent = MyDearAgent()
    agent.run()
