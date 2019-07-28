# reference code https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/a3c/worker.py
# https://github.com/vietnguyen91/Super-mario-bros-A3C-pytorch
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
from logger import TensorboardLogger
import random

use_cuda = torch.cuda.is_available()
seed = 11037
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


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
        self.save_dir = './model/mario'
        print("update freq:",self.update_freq)
        print("n_processes:",self.n_processes)
        print("display freq:", self.display_freq)
        print("save freq:", self.save_freq)
        print("save dir:", self.save_dir)

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        
        self.envs = env
        if self.envs == None:
            self.envs = make_vec_envs('SuperMarioBros-v0', self.seed,
                    self.n_processes)
        
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        print("using device:", self.device)

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
   
        self.tensorboard = TensorboardLogger("./log/mario_log")
        self.step_s = 0
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
        obs_batch = (self.rollouts.obs[:-1]).view(-1, self.obs_shape[0], self.obs_shape[1], self.obs_shape[2]) # (n_step*n_processes=80, 4, 84, 84)
        hidden_batch = (self.rollouts.hiddens[0]).view(self.n_processes, self.hidden_size) # (n_step*n_processes=80, hidden_size=512)
        mask_batch = (self.rollouts.masks[:-1]).view(-1, 1) # (n_step*n_processes=80, 1)
        if (len(obs_batch.shape) != 4 or len(hidden_batch.shape) != 2) or len(mask_batch.shape) !=2:
            raise NotImplementedError("shape error")

        # print("obs batch size", obs_batch.size())
        # print("hidden batch size", hidden_batch.size())
        # print("mask batch size", mask_batch.size())
        obs_batch = obs_batch.to(self.device)        
        hidden_batch = hidden_batch.to(self.device)
        mask_batch = mask_batch.to(self.device)
        values, action_probs, hiddens = self.model(obs_batch, hidden_batch, mask_batch)
        
        log_probs = torch.log(action_probs).view(self.update_freq*self.n_processes,-1)
        action_gather = self.rollouts.actions.view(self.update_freq*self.n_processes,-1)
        log_action_probs = log_probs.gather(1, action_gather)        
        log_action_probs = log_action_probs.view(self.update_freq, self.n_processes, -1)
        #dist = torch.distributions.Categorical(action_probs.view(self.update_freq,self.n_processes,-1))
        #log_action_probs = dist.log_prob(self.rollouts.actions)


        with torch.no_grad():
            self.model.eval()
            obs_next_batch = (self.rollouts.obs[1:self.update_freq+1]).view(-1, self.obs_shape[0], self.obs_shape[1], self.obs_shape[2]) # (n_step + 1, n_processes, 4, 84, 84)
            hidden_next_batch = (self.rollouts.hiddens[1]).view(self.n_processes, self.hidden_size) # (n_step + 1, n_processes, hidden_size)
            mask_next_batch = (self.rollouts.masks[1:self.update_freq+1]).view(-1, 1) # (n_step + 1*n_processes, 1)
            if (len(obs_next_batch.shape) != 4 or len(hidden_next_batch.shape) != 2) or len(mask_next_batch.shape) !=2:
                raise NotImplementedError("shape error")
            #print("obs next batch size", obs_next_batch.size())
            #print("hidden next batch size", hidden_next_batch.size())
            #print("mask next batch size", mask_next_batch.size())
            obs_next_batch = obs_next_batch.to(self.device)
            hidden_next_batch = hidden_next_batch.to(self.device)
            mask_next_batch = mask_next_batch.to(self.device)
            next_values, _, _ = self.model(obs_next_batch, hidden_next_batch, mask_next_batch)
        
        

        # next_values[t] = V(s)_t
        
        # next feedforward
        #[0,1,2,3,4,5]
        # obs = [ob(0), ob(1), ob(2), ob(3), ob(4), ob(5)]
        # mask = [m(0),m(1),m(2),m(3),m(4),m(5)]
        # values = [v(0),v(1),v(2),v(3),v(4),v(5)]
        # self.rollouts.value_preds = [v(0), v(1), v(2), v(3), v(4)]
        # TODO:
        # calculate
        # target = f(values, next_values)
        # value_loss = mse(values, target)
        #values = self.rollouts.value_preds[0 : self.update_freq]
        self.model.train()
        target_y = self.rollouts.rewards + self.gamma * next_values.view(self.update_freq,self.n_processes,1) # [v(1),v(2),v(3),v(4),v(5)]
        value_loss_fn = torch.nn.MSELoss()
        value_loss = 0.5 * value_loss_fn(values, target_y.view(self.update_freq*self.n_processes,-1))
        # TODO:
        # action_loss = log(...) * advantage_function
        advantage_function = R - (values.view(self.update_freq,self.n_processes,-1)).detach()
        action_loss = (-log_action_probs*advantage_function).mean()
        self.tensorboard.histogram_summary("log_action_prob", log_action_probs)
        self.tensorboard.histogram_summary("advantage function", advantage_function)
        #entropy = dist.entropy()
        loss = action_loss + value_loss 
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.grad_norm)
        self.optimizer.step()
        # TODO:
        # Clear rollouts after update (RolloutStorage.reset())
        self.rollouts.reset()
        self.tensorboard.scalar_summary("action_loss", action_loss.item())
        self.tensorboard.scalar_summary("value_loss", value_loss.item())
        self.tensorboard.scalar_summary("loss",loss.item())
        return loss.item()

    def _step(self, obs, hiddens, masks):
        """ 
        Input : obs = (n_processes,4,84,84)
                hiddens = (16,512)
                masks = (16,1)
        """
        with torch.no_grad():
        # TODO:
        # Sample actions from the output distributions
        # HINT: you can use torch.distributions.Categorical
            values, action_probs, hiddens_next = self.make_action(obs, hiddens, masks)
            dist = torch.distributions.Categorical(action_probs)
            actions = dist.sample()
            self.tensorboard.scalar_summary("chosen_action",actions.item(),self.step_s)
            self.step_s += 1
        #log_action_prob = torch.unsqueeze(dist.log_prob(actions),1) # expand dim
        #entropy = dist.entropy()

        obs_next, rewards, dones, infos = self.envs.step(actions.cpu().numpy()) # synchronously step
        masks_next = torch.from_numpy(1 - np.expand_dims(dones,axis=1)) # (16,1)
        actions = torch.unsqueeze(actions, 1)
        rewards = torch.from_numpy(np.expand_dims(rewards,axis=1))
        # TODO:
        # Store transitions (obs, hiddens, actions, values, rewards, masks)
        # You need to convert arrays to tensors first
        # HINT: masks = (1 - dones)
        self.rollouts.insert(torch.from_numpy(obs_next), 
                            hiddens_next, 
                            actions, 
                            values,  
                            rewards, 
                            masks_next)

    def train(self):
        print('Start training')
        running_reward = deque(maxlen=10)
        episode_rewards = torch.zeros(self.n_processes, 1).to(self.device)
        total_steps = 0
        best_avg_reward = float('-inf')
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
            
            
            loss_term = self._update()

            total_steps += self.update_freq * self.n_processes
            # Log & save model
            if len(running_reward) == 0:
                avg_reward = 0
            else:
                avg_reward = sum(running_reward) / len(running_reward)

            if total_steps % self.display_freq == 0:
                print('Steps: %d/%d | Avg reward: %f'%
                        (total_steps, self.max_steps, avg_reward))
                self.tensorboard.scalar_summary("reward",avg_reward)
            
            if total_steps % self.save_freq == 0:
                print("*****save model*****")
                self.save_model('mario_model.pt')
            
            if avg_reward > best_avg_reward:
                print("#####save best model#####")
                best_avg_reward = avg_reward
                self.save_model('mario_best.cpt')

            if total_steps >= self.max_steps:
                break

            self.tensorboard.update()
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
            self.model.eval()
            obs = obs.to(self.device)
            hiddens = hiddens.to(self.device)
            masks = masks.to(self.device)
            values, action_probs, hiddens = self.model(obs, hiddens, masks)
            return values, action_probs, hiddens  
        else :
            pass
 
