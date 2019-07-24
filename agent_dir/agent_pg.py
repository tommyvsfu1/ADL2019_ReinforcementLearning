import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent_dir.agent import Agent
from environment import Environment
from logger import TensorboardLogger

seed = 11037
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.action_prob = []
        self.values = []
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.action_prob[:]
        del self.values[:]
def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_num, hidden_dim):
        super(PolicyNet, self).__init__()
        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_num),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        action_prob = self.action_layer(x)
        return action_prob

class AgentPG(Agent):
    def __init__(self, env, args):
        self.env = env
        self.model = PolicyNet(state_dim = self.env.observation_space.shape[0],
                               action_num= self.env.action_space.n,
                               hidden_dim=64)
        
        
        if args.test_pg:
            self.load('pg.cpt')
        # discounted reward
        self.gamma = 0.99 
        
        # training hyperparameters
        self.num_episodes = 4000 # total training episodes (actually too large...)
        self.display_freq = 10 # frequency to display training progress
        
        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3)
        
        # saved rewards and actions
        self.memory = Memory()
    
        # log
        self.tensorboard = TensorboardLogger('./pg_lunar')

        # device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else :
            self.device = torch.device('cpu')

    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.model.state_dict(), save_path)
    
    def load(self, load_path):
        print('load model from', load_path)
        self.model.load_state_dict(torch.load(load_path))

    def init_game_setting(self):
        pass
    def make_action(self, state, test=False):
        # TODO:
        # Use your model to output distribution over actions and sample from it.
        # HINT: google torch.distributions.Categorical
        if not test:
            state_tensor = (torch.from_numpy(state).unsqueeze(0)).to(self.device)
            prob = self.model(state_tensor)
            dist = torch.distributions.Categorical(prob)
            try :
                action = dist.sample()
            except:
                print("error sample")
                print("prob", prob)
            self.memory.logprobs.append(dist.log_prob(action))

        if test:
            # stochastic 
            self.model.eval()
            state_tensor = (torch.from_numpy(state).unsqueeze(0)).to(self.device)
            prob = self.model(state_tensor)
            dist = torch.distributions.Categorical(prob)
            action = dist.sample()

        return action.item()

    def update(self):
        self.model.train()  
        # TODO:
        # discount your saved reward
        R = 0
        advantage_function = []
        
        for t in reversed(range(0, len(self.memory.rewards))):
            R = R * self.gamma + self.memory.rewards[t]
            advantage_function.insert(0, R)

        # turn rewards to pytorch tensor and standardize
        advantage_function = torch.Tensor(advantage_function).to(self.device)
        advantage_function = (advantage_function - advantage_function.mean()) / (advantage_function.std() + np.finfo(np.float32).eps)

        # TODO:      
        #x = torch.stack(self.memory.states).float().to(self.device)
        #action_tensor = torch.stack(self.memory.actions).to(self.device)


        policy_loss = []
        for log_prob, reward in zip(self.memory.logprobs, advantage_function):
            policy_loss.append(-log_prob * reward)
        
        # Update network weights
        self.optimizer.zero_grad()
        loss = torch.cat(policy_loss).sum()
        loss.backward()
        self.optimizer.step()   
        
        

    def train(self):
        avg_reward = None # moving average of reward
        best_avg_reward = float('-inf')
        for epoch in range(self.num_episodes):
            state = self.env.reset()
            self.init_game_setting()
            done = False
            while(not done):
                self.memory.states.append(torch.from_numpy(state))
                action = self.make_action(state)
                state, reward, done, _ = self.env.step(action)
                
                self.memory.actions.append(torch.tensor([action]))
                self.memory.rewards.append(reward)                
            # for logging 
            last_reward = np.sum(self.memory.rewards)
            avg_reward = last_reward if not avg_reward else avg_reward * 0.9 + last_reward * 0.1
            self.tensorboard.scalar_summary("avg_reward", avg_reward)

            # update model, tensorboard
            self.update()
            self.tensorboard.update()

            # update tensorboard
            if epoch % self.display_freq == 0:
                print('Epochs: %d/%d | Avg reward: %f '%
                       (epoch, self.num_episodes, avg_reward))
            
            if avg_reward > 50: # to pass baseline, avg. reward > 50 is enough.
                self.save('pg_baseline.cpt')

            # save the best model
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                self.save('pg_best.cpt')

            self.memory.clear_memory()