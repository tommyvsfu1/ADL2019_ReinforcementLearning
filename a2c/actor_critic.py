import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ActorCritic(nn.Module):
    def __init__(self, obs_shape, act_shape, hidden_size, recurrent):
        super(ActorCritic, self).__init__()
        
        self.recurrent = recurrent
        self.hidden_size = hidden_size
        # obs_shape = (4, 84, 84)
        self.head = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(32 * 7 * 7, hidden_size),
            nn.ReLU()
        )

        if self.recurrent:
            self.rnn =  nn.GRU(hidden_size, hidden_size)
        
        self.actor = nn.Linear(hidden_size, act_shape)
        self.critic = nn.Linear(hidden_size, 1)
        self.flatten = Flatten()
        self.reset_parameters()

    def reset_parameters(self):
        
        def _weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1 or classname.find('Linear') != -1:
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.head.apply(_weights_init)

        if self.recurrent:
            for name, param in self.rnn.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)
        
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight)
        nn.init.constant_(self.actor.bias, 0)
        nn.init.constant_(self.critic.bias, 0)

    def _forward_rnn(self, x, hiddens, masks):
        '''
        Inputs:
            x: observations : (n_steps * n_processes, hidden_size)
            hiddens: hidden states of 1st step : (n_processes, hidden_size)
            masks: whether to reset hidden state : (n_steps * n_processes, 1)
        Returns:
            x: outputs of RNN : (n_steps * n_processes, hidden_size)
            hiddens: hidden states of last step : (n_processes, hidden_size)
        '''

        # print("x size", x.size())
        # print("hidden size", hiddens.size())
        # print("mask size", masks.size())   
        ### single case:
        # x = 16,512
        # hiddens = 16,512
        # masks = 16,1
        ### batch case:
        # x = 80,512
        # h = 16,512
        # masks = 80,1

        n_processes = hiddens.shape[0] # 16
        hidden_size = hiddens.shape[1] # 512
        n_steps = masks.shape[0] // n_processes # 80/16 = 5
        if (n_steps != 1) and (n_steps != 5):
            raise NotImplementedError("size wrong")        

        # TODO 
        # step 1: Unflatten the tensors to (n_steps, n_processes, -1)
        x_input = x.view(n_steps, n_processes, -1) # 5,16,512
        mask_input = masks.view(n_steps, n_processes, -1) # 5,16,1

        # print("x_input size", x_input.size())
        # print("mask input size", mask_input.size())
        # step 2: Run a for loop through time to forward rnn
        # rnn propogating network
        #        y  
        #        |
        # h0 -> GRU -> h1
        #        |
        #        x
        y = torch.zeros_like(x_input) # (n_step, 16, 512)
        h_t = hiddens # (16,512)
        for t in range(n_steps):             
            x_t = x_input[t].unsqueeze(0) # (16,512) -> (1,16,512)
            mask_t = mask_input[t] # (16,1)
            h_t = (h_t * mask_t).view(-1,n_processes,hidden_size)
            y[t], h_t = self.rnn(x_t, h_t)
            # Debug
            #print("x_t size", x_t.size())
            #print("mask_t size", mask_t.size())
            #print("h_t size", h_t.size())

        # step 3: Flatten the outputs
        y = y.view(n_steps*n_processes, hidden_size)
        hiddens_out = h_t.view(n_processes, hidden_size)
        # HINT: You must set hidden states to zeros when masks == 0 in the loop 
        
        return y, hiddens_out

    def forward(self, inputs, hiddens, masks):
        x = self.head(inputs / 255.0)
        if self.recurrent:
            x, hiddens = self._forward_rnn(x, hiddens, masks)
        
        values = self.critic(x)
        action_probs = self.actor(x)
        action_probs = F.softmax(action_probs, -1)
        
        return values, action_probs, hiddens

# Test
# ac = ActorCritic(obs_shape=(4, 84, 84),act_shape=12,hidden_size=512,recurrent=True)
# a = torch.ones((80,4,84,84))
# h = torch.ones((16,512))
# m = torch.ones((80,1))
# ac(a,h,m)