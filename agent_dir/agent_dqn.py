import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from agent_dir.agent import Agent
from environment import Environment
from collections import namedtuple
from logger import TensorboardLogger

use_cuda = torch.cuda.is_available()
seed = 11037
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

Transition = namedtuple('Transition',
                        ('state', 'action','reward', 'next_state'))
def expand_dim(x):
    y = torch.unsqueeze(input=x,dim=0)
    return y

class ReplayBuffer(object):
    """
    Replay Buffer for Q function
        default size : 20000 of (s_t, a_t, r_t, s_t+1)
    """
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """
        Push (s_t, a_t, r_t, s_t+1) into buffer
            Input : s_t, a_t, r_t, s_t+1
            Output : None
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Dueling_DQN(nn.Module):
    '''
    This architecture is the one from OpenAI Baseline, with small modification.
    '''
    def __init__(self, channels, num_actions):
        super(Dueling_DQN, self).__init__()
        self.num_actions = num_actions
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(3136, 512)
        self.head = nn.Linear(512, num_actions)
        self.fc1_val = torch.nn.Linear((512), 1)
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.01)

        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.lrelu(self.fc(x.view(x.size(0), -1)))
        adv = self.head(x)
        val = self.fc1_val(x)
        Q = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0),self.num_actions)
        return Q

class DQN(nn.Module):
    '''
    This architecture is the one from OpenAI Baseline, with small modification.
    '''
    def __init__(self, channels, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(3136, 512)
        self.head = nn.Linear(512, num_actions)
        
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.01)

        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.lrelu(self.fc(x.view(x.size(0), -1)))
        q = self.head(x)
        return q

class AgentDQN(Agent):
    def __init__(self, env, args):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else :
            self.device = torch.device('cpu')

        print("using device", self.device)

        self.env = env
        self.env.seed(seed)
        self.input_channels = 4
        self.num_actions = self.env.action_space.n
        # TODO:
        # Initialize your replay buffer

        # build target, online network
        if args.model_use == "Dueling":
            self.target_net = Dueling_DQN(self.input_channels, self.num_actions)
            self.online_net = Dueling_DQN(self.input_channels, self.num_actions)
        else :
            self.target_net = DQN(self.input_channels, self.num_actions)
            self.online_net = DQN(self.input_channels, self.num_actions)

        self.target_net = self.target_net.to(self.device)
        self.online_net = self.online_net.to(self.device)

        if args.test_dqn:
            self.load('dqn_best')
        
        # discounted reward
        self.GAMMA = 0.99 
        
        # training hyperparameters
        self.train_freq = 4 # frequency to train the online network
        self.learning_start = 10000 # before we start to update our network, we wait a few steps first to fill the replay.
        self.batch_size = 32
        self.num_timesteps = 3000000 # total training steps
        self.display_freq = 10 # frequency to display training progress
        self.save_freq = 200000 # frequency to save the model
        self.target_update_freq = 1000 # frequency to update target network
        self.epsilon = 0
        self.replay_buffer = ReplayBuffer()
        self.EPS_DECAY = 80000
        self.tensorboard = TensorboardLogger(dir='./log/dueling_dqn_assault')

        # optimizer
        self.optimizer = optim.RMSprop(self.online_net.parameters(), lr=1e-4)

        self.steps = 0 # num. of passed steps. this may be useful in controlling exploration


    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.online_net.state_dict(), save_path + '_online.cpt')
        torch.save(self.target_net.state_dict(), save_path + '_target.cpt')

    def load(self, load_path):
        print('load model from', load_path)
        if use_cuda:
            self.online_net.load_state_dict(torch.load(load_path + '_online.cpt'))
            self.target_net.load_state_dict(torch.load(load_path + '_target.cpt'))
        else:
            self.online_net.load_state_dict(torch.load(load_path + '_online.cpt', map_location=lambda storage, loc: storage))
            self.target_net.load_state_dict(torch.load(load_path + '_target.cpt', map_location=lambda storage, loc: storage))

    def init_game_setting(self):
        # we don't need init_game_setting in DQN
        pass
    
    def make_action(self, state, test=False):
        if not test:
            # TODO:
            # At first, you decide whether you want to explore the environemnt
            eps = 0.01 + (0.9 - 0.01) * math.exp(-1. * self.steps / self.EPS_DECAY)
            # TODO:
            # if explore, you randomly samples one action
            # else, use your model to predict action
            if random.random() > eps:
                Q_s_a= self.online_net(state)
                action = torch.argmax(Q_s_a)
                return action.item()
            else :
                action = self.env.get_random_action()
                return action
        else :
            self.online_net.eval()
            state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0)
            Q_s_a = self.online_net(state)
            action = torch.argmax(Q_s_a)
            return action.item()
    def update(self):
        # TODO:
        # To update model, we sample some stored experiences as training examples.
        if len(self.replay_buffer) < self.batch_size:
            return 
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        # next_state_batch = torch.cat(batch.state).to(self.device)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        
        # TODO:
        # Compute Q(s_t, a) with your model.
        state_action_values = self.online_net(state_batch).gather(1, action_batch).reshape(-1)

        with torch.no_grad():
            # TODO:
            # Compute Q(s_{t+1}, a) for all next states.
            # Since we do not want to backprop through the expected action values,
            # use torch.no_grad() to stop the gradient from Q(s_{t+1}, a)
            # next_state_values= self.target_net(next_state_batch).max(1,keepdim=True)[0]
            next_state_values = torch.zeros(self.batch_size, device=self.device)
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        # TODO:
        # Compute the expected Q values: rewards + gamma * max(Q(s_{t+1}, a))
        # You should carefully deal with gamma * max(Q(s_{t+1}, a)) when it is the terminal state.
        expected_state_action_values = self.GAMMA * (next_state_values) + reward_batch.reshape(-1)

        # TODO:
        # Compute temporal difference loss
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self):
        episodes_done_num = 0 # passed episodes
        total_reward = 0 # compute average reward
        best_avg_reward = float('-inf')
        loss = 0 
        while(True):
            state = self.env.reset()
            # State: (84,84,4) --> (1,4,84,84)
            state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0)
            state = state.to(self.device)

            done = False
            while(not done):
                # select and perform action
                action = self.make_action(state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward

                # process new state
                next_state = torch.from_numpy(next_state).permute(2,0,1).unsqueeze(0)
                next_state = next_state.to(self.device)
                if done:
                    next_state = None

                # TODO:
                # store the transition in memory
                a_0 = torch.tensor([action], device=self.device)
                r_0 = torch.tensor([reward], device=self.device)
                self.replay_buffer.push(state, expand_dim(a_0), expand_dim(r_0),next_state)

                # move to the next state
                state = next_state

                # Perform one step of the optimization
                if self.steps > self.learning_start and self.steps % self.train_freq == 0:
                    loss = self.update()

                # update target network
                if self.steps > self.learning_start and self.steps % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())

                # save the model
                if self.steps % self.save_freq == 0:
                    self.save('dqn')

                self.steps += 1
                self.tensorboard.update()

            if episodes_done_num % self.display_freq == 0:
                avg_reward = total_reward / self.display_freq
                print('Episode: %d | Steps: %d/%d | Avg reward: %f | loss: %f '%
                        (episodes_done_num, self.steps, self.num_timesteps, avg_reward, loss))
                self.tensorboard.scalar_summary("Avg reward", total_reward / self.display_freq)
                self.tensorboard.scalar_summary("loss", loss)
                total_reward = 0

                # save the best model
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    self.save('dqn_best')



            episodes_done_num += 1
            if self.steps > self.num_timesteps:
                break
        self.save('dqn')
