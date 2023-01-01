import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch import from_numpy, no_grad, save, load, tensor, clamp
from torch import float as torch_float
from torch import long as torch_long
from torch import min as torch_min
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
from torch import manual_seed
from collections import namedtuple

Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])


class PPOAgent:
    """
    PPOAgent implements the PPO RL algorithm (https://arxiv.org/abs/1707.06347).
    It works with a set of discrete actions.
    It uses the Actor and Critic neural network classes defined below.
    """

    def __init__(self, number_of_inputs, number_of_actor_outputs, clip_param=0.2, max_grad_norm=0.5, ppo_update_iters=5,
                 batch_size=8, gamma=0.99, use_cuda=False, actor_lr=0.001, critic_lr=0.003, seed=None):
        super().__init__()
        if seed is not None:
            manual_seed(seed)

        # Hyper-parameters
        self.clip_param = clip_param
        self.max_grad_norm = max_grad_norm
        self.ppo_update_iters = ppo_update_iters
        self.batch_size = batch_size
        self.gamma = gamma
        self.use_cuda = use_cuda

        # models
        self.actor_net = Actor(number_of_inputs, number_of_actor_outputs)
        self.critic_net = Critic(number_of_inputs)

        if self.use_cuda:
            self.actor_net.cuda()
            self.critic_net.cuda()

        # Create the optimizers
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), actor_lr)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), critic_lr)

        # Training stats
        self.buffer = []

    def work(self, agent_input, type_="selectAction"):
        """
        Forward pass of the PPO agent. Depending on the type_ argument, it either explores by sampling its actor's
        softmax output, or eliminates exploring by selecting the action with the maximum probability (argmax).

        :param agent_input: The actor neural network input vector
        :type agent_input: vector
        :param type_: "selectAction" or "selectActionMax", defaults to "selectAction"
        :type type_: str, optional
        """
        agent_input = from_numpy(np.array(agent_input)).float().unsqueeze(0)  # Add batch dimension with unsqueeze

        if self.use_cuda:
            agent_input = agent_input.cuda()

        with no_grad():
            action_prob = self.actor_net(agent_input)

        if type_ == "selectAction":
            c = Categorical(action_prob)
            action = c.sample()
            return action.item(), action_prob[:, action.item()].item()
        elif type_ == "selectActionMax":
            return np.argmax(action_prob).item(), 1.0

    def save(self, path):
        """
        Save actor and critic models in the path provided.

        :param path: path to save the models
        :type path: str
        """
        save(self.actor_net.state_dict(), path + '_actor.pkl')
        save(self.critic_net.state_dict(), path + '_critic.pkl')

    def load(self, path):
        """
        Load actor and critic models from the path provided.

        :param path: path where the models are saved
        :type path: str
        """
        actor_state_dict = load(path + '_actor.pkl')
        critic_state_dict = load(path + '_critic.pkl')
        self.actor_net.load_state_dict(actor_state_dict)
        self.critic_net.load_state_dict(critic_state_dict)

    def store_transition(self, transition):
        """
        Stores a transition in the buffer to be used later.

        :param transition: contains state, action, action_prob, reward, next_state
        :type transition: namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])
        """
        self.buffer.append(transition)

    def train_step(self, batch_size=None):
        """
        Performs a training step for the actor and critic models, based on transitions gathered in the
        buffer. It then resets the buffer.

        :param batch_size: Overrides agent set batch size, defaults to None
        :type batch_size: int, optional
        """
        # Default behaviour waits for buffer to collect at least one batch_size of transitions
        if batch_size is None:
            if len(self.buffer) < self.batch_size:
                return
            batch_size = self.batch_size

        # Extract states, actions, rewards and action probabilities from transitions in buffer
        state = tensor([t.state for t in self.buffer], dtype=torch_float)
        action = tensor([t.action for t in self.buffer], dtype=torch_long).view(-1, 1)
        reward = [t.reward for t in self.buffer]
        old_action_log_prob = tensor([t.a_log_prob for t in self.buffer], dtype=torch_float).view(-1, 1)

        # Unroll rewards
        reward_unroll = 0
        gain = []
        for r in reward[::-1]:
            reward_unroll = r + self.gamma * reward_unroll
            gain.insert(0, reward_unroll)
        gain = tensor(gain, dtype=torch_float)

        # Send everything to cuda if used
        if self.use_cuda:
            state, action, old_action_log_prob = state.cuda(), action.cuda(), old_action_log_prob.cuda()
            gain = gain.cuda()

        # Repeat the update procedure for ppo_update_iters
        for i in range(self.ppo_update_iters):
            # Create randomly ordered batches of size batch_size from buffer
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), batch_size, False):
                # Calculate the advantage at each step
                gain_index = gain[index].view(-1, 1)
                value = self.critic_net(state[index])
                delta = gain_index - value
                advantage = delta.detach()

                # Get the current probabilities
                # Apply past actions with .gather()
                action_prob = self.actor_net(state[index]).gather(1, action[index])  # new policy

                # PPO
                ratio = (action_prob / old_action_log_prob[index])  # Ratio between current and old policy probabilities
                surr1 = ratio * advantage
                surr2 = clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch_min(surr1, surr2).mean()  # MAX->MIN descent
                self.actor_optimizer.zero_grad()  # Delete old gradients
                action_loss.backward()  # Perform backward step to compute new gradients
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)  # Clip gradients
                self.actor_optimizer.step()  # Perform training step based on gradients

                # update critic network
                value_loss = F.mse_loss(gain_index, value)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()

        # After each training step, the buffer is cleared
        del self.buffer[:]


class Actor(nn.Module):
    def __init__(self, number_of_inputs, numberOfOutputs):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(number_of_inputs, 10)
        self.fc2 = nn.Linear(10, 10)
        self.action_head = nn.Linear(10, numberOfOutputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self, number_of_inputs):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(number_of_inputs, 10)
        self.fc2 = nn.Linear(10, 10)
        self.state_value = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.state_value(x)
        return value
