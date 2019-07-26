import torch
from torch.distributions import Categorical
from torch.optim import RMSprop
from torch.nn.utils import clip_grad_norm_

from a2c.environment_a2c import make_vec_envs
from a2c.storage import RolloutStorage
from a2c.actor_critic import ActorCritic

from collections import deque
import os
import numpy as np
use_cuda = torch.cuda.is_available()

class AgentMario:
    def __init__(self, env, args):

        # Hyperparameters
        self.lr = 7e-4
        self.gamma = 0.9
        self.hidden_size = 512
        self.update_freq = 5
        self.n_processes = 16
        self.seed = 7122
        self.max_steps = 1e7
        self.grad_norm = 0.5
        self.entropy_weight = 0.05

        #######################    NOTE: You need to implement
        self.recurrent = True # <- ActorCritic._forward_rnn()
        #######################    Please check a2c/actor_critic.py
        
        self.display_freq = 4000
        self.save_freq = 100000
        self.save_dir = './checkpoints/'

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        
        self.envs = env
        if self.envs == None:
            self.envs = make_vec_envs('SuperMarioBros-v0', self.seed,
                    self.n_processes)
        self.device = torch.device("cuda:0" if use_cuda else "cpu")

        self.obs_shape = self.envs.observation_space.shape
        self.act_shape = self.envs.action_space.n

        self.rollouts = RolloutStorage(self.update_freq, self.n_processes,
                self.obs_shape, self.act_shape, self.hidden_size) 
        self.model = ActorCritic(self.obs_shape, self.act_shape,
                self.hidden_size, self.recurrent).to(self.device)
        self.optimizer = RMSprop(self.model.parameters(), lr=self.lr, 
                eps=1e-5)

        self.hidden = None
        self.init_game_setting()
   
    def _update(self):
        # TODO: Compute returns
        # R_t = reward_t + gamma * R_{t+1}
        # reward shape = (n_steps, n_processes, 1)
        R = torch.zeros_like(self.rollouts.rewards) # (n_step, n_processes, 1) 
        for t in reversed(range(0, self.update_freq)):
            R[t] = R[t] * self.gamma + self.rollouts.rewards[t]
        # TODO:
        # Compute actor critic loss (value_loss, action_loss)
        # OPTIONAL: You can also maxmize entropy to encourage exploration
        # loss = value_loss + action_loss (- entropy_weight * entropy)

        #  Feedforward (do not use with no_grad() because we need Backprop)
        obs_batch = (self.rollouts.obs).view(-1, self.obs_shape[0], self.obs_shape[1], self.obs_shape[2]) # (n_step + 1, n_processes, 4, 84, 84)
        hidden_batch = (self.rollouts.hiddens).view(-1, self.hidden_size) # (n_step + 1, n_processes, hidden_size)
        mask_batch = (self.rollouts.masks).view(-1, 1) # (n_step + 1, n_processes, 1)
        next_values = self.rollouts.value_preds

        values, action_probs, hiddens = self.model(obs_batch, hidden_batch, mask_batch)

        # TODO:
        # calculate
        # target = f(values, next_values)
        # value_loss = mse(values, target)


        # TODO:
        # action_loss = log(...) * advantage_function


        
        print("values shape", values.size())
        print("values", (values.view(6,16,1))[:,0,0])
        print("next values", next_values[:,0,0])
        entropy = (torch.distributions.Categorical(action_probs)).entropy()
        loss = - 0.01 * entropy
        # Update
        #self.optimizer.zero_grad()
        #loss.backward()
        #clip_grad_norm_(self.model.parameters(), self.grad_norm)
        #self.optimizer.step()
        
        # TODO:
        # Clear rollouts after update (RolloutStorage.reset())

        #return loss.item()

    def _step(self, obs, hiddens, masks):
        with torch.no_grad():
            # TODO:
            # Sample actions from the output distributions
            # HINT: you can use torch.distributions.Categorical
            values, action_probs, hiddens = self.make_action(obs, hiddens, masks)
            dist = torch.distributions.Categorical(action_probs)
            actions = dist.sample()

        obs, rewards, dones, infos = self.envs.step(actions.cpu().numpy()) # synchronously step
        masks = torch.from_numpy(1 - np.expand_dims(dones,axis=1))
        actions = torch.unsqueeze(actions, 1)
        rewards = torch.from_numpy(np.expand_dims(rewards,axis=1))
        # TODO:
        # Store transitions (obs, hiddens, actions, values, rewards, masks)
        # You need to convert arrays to tensors first
        # HINT: masks = (1 - dones)
        self.rollouts.insert(torch.from_numpy(obs), 
                            hiddens, 
                            actions, 
                            values,  
                            rewards, 
                            masks)

    def train(self):
        print('Start training')
        running_reward = deque(maxlen=10)
        episode_rewards = torch.zeros(self.n_processes, 1).to(self.device)
        total_steps = 0
        
        # Store first observation
        obs = torch.from_numpy(self.envs.reset()).to(self.device)
        self.rollouts.obs[0].copy_(obs)
        self.rollouts.to(self.device) 
        while True:
            # Update once every n-steps
            for step in range(self.update_freq):
                self._step(
                    self.rollouts.obs[step],
                    self.rollouts.hiddens[step],
                    self.rollouts.masks[step])

                # Calculate episode rewards
                episode_rewards += self.rollouts.rewards[step]
                for r, m in zip(episode_rewards, self.rollouts.masks[step + 1]): # n_processes times
                    if m == 0:
                        running_reward.append(r.item()) # when done, update running reward
                episode_rewards *= self.rollouts.masks[step + 1] # episode reward continues or back to 0
            loss = self._update()
            total_steps += self.update_freq * self.n_processes
            print("self .rollout step", self.rollouts.step)
            # Log & save model
            if len(running_reward) == 0:
                avg_reward = 0
            else:
                avg_reward = sum(running_reward) / len(running_reward)

            if total_steps % self.display_freq == 0:
                print('Steps: %d/%d | Avg reward: %f'%
                        (total_steps, self.max_steps, avg_reward))
            
            if total_steps % self.save_freq == 0:
                self.save_model('model.pt')
            
            if total_steps >= self.max_steps:
                break

    def save_model(self, filename):
        torch.save(self.model, os.path.join(self.save_dir, filename))

    def load_model(self, path):
        self.model = torch.load(path)

    def init_game_setting(self):
        if self.recurrent:
            self.hidden = torch.zeros(1, self.hidden_size).to(self.device)

    def make_action(self, obs, hiddens, masks, test=False):
        # TODO: Use you model to choose an action
        if not test:
            values, action_probs, hiddens = self.model(obs, hiddens, masks)
        else :
            raise NotImplementedError("Not Implement Test case")
        return values, action_probs, hiddens   
