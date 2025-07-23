import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class OUActionNoise(object):
    """
    Ornstein–Uhlenbeck noise, https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
    """

    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.x_prev = None
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, output_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, *output_shape))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        reward = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, reward, new_states, terminal


class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, lr, fc1_dims, fc2_dims, fc3_dims, name,
                 chkpt_dir='./models/saved/default_ddpg/', use_cuda=False):
        super(CriticNetwork, self).__init__()

        self.input_shape = input_shape
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims

        self.output_shape = output_shape
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_ddpg')

        self.fc1 = nn.Linear(*self.input_shape, self.fc1_dims)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.action_value = nn.Linear(*self.output_shape, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims + fc2_dims, fc3_dims)
        self.q = nn.Linear(fc3_dims, 1)

        self.initialization()

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        if use_cuda:
            self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu'

        self.to(self.device)

    def initialization(self):
        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('leaky_relu'))

        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('leaky_relu'))

        nn.init.xavier_uniform_(self.action_value.weight, gain=nn.init.calculate_gain('leaky_relu'))

        nn.init.xavier_uniform_(self.fc3.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = F.leaky_relu(state_value)
        # state_value = self.bn1(state_value)

        state_value = self.fc2(state_value)
        state_value = F.leaky_relu(state_value)
        # state_value = self.bn2(state_value)

        action_value = self.action_value(action)
        action_value = F.leaky_relu(action_value)

        state_action_value = T.cat((action_value, state_value), dim=1)
        state_action_value = self.fc3(state_action_value)
        state_action_value = F.relu(state_action_value)

        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self):
        print("...saving checkpoint....")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("..loading checkpoint...")
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, lr, fc1_dims, fc2_dims, fc3_dims, name,
                 chkpt_dir='./models/saved/default_ddpg/', use_cuda=False):
        super(ActorNetwork, self).__init__()
        self.lr = lr
        self.input_shape = input_shape
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims

        self.output_shape = output_shape
        self.checkpoint_file = os.path.join(chkpt_dir, name + "_ddpg")
        self.fc1 = nn.Linear(*self.input_shape, self.fc1_dims)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.bn3 = nn.LayerNorm(self.fc3_dims)

        self.mu = nn.Linear(self.fc3_dims, *self.output_shape)

        self.initialization()

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        if use_cuda:
            self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu'

        self.to(self.device)

    def initialization(self):
        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('leaky_relu'))

        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('leaky_relu'))

        nn.init.xavier_uniform_(self.fc3.weight, gain=nn.init.calculate_gain('leaky_relu'))

        nn.init.xavier_uniform_(self.mu.weight, gain=nn.init.calculate_gain('tanh'))

    def forward(self, state):
        x = self.fc1(state)
        x = F.leaky_relu(x)
        # x = self.bn1(x)

        x = self.fc2(x)
        x = F.leaky_relu(x)
        # x = self.bn2(x)

        x = self.fc3(x)
        x = F.leaky_relu(x)
        # x = self.bn3(x)

        x = T.tanh(self.mu(x))

        return x

    def save_checkpoint(self):
        print("...saving checkpoint....")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("..loading checkpoint...")
        self.load_state_dict(T.load(self.checkpoint_file))


class DDPGAgent(object):
    def __init__(self, input_shape, output_shape, lr_actor=0.0001, lr_critic=0.001, tau=0.01, gamma=0.99,
                 max_size=1000000, layer1_size=10, layer2_size=20, layer3_size=10, batch_size=8):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.memory = ReplayBuffer(max_size, input_shape, output_shape)
        self.output_shape = output_shape

        self.actor = ActorNetwork(input_shape, output_shape, lr_actor, layer1_size, layer2_size, layer3_size,
                                  name="Actor")

        self.target_actor = ActorNetwork(input_shape, output_shape, lr_actor, layer1_size, layer2_size, layer3_size,
                                         name="TargetActor")

        self.critic = CriticNetwork(input_shape, output_shape, lr_critic, layer1_size, layer2_size, layer3_size,
                                    name="Critic")

        self.target_critic = CriticNetwork(input_shape, output_shape, lr_critic, layer1_size, layer2_size, layer3_size,
                                           name="TargetCritic")

        self.noise = OUActionNoise(np.zeros(output_shape))

        self.update_network_parameters(tau=tau)

    def choose_action_train(self, observation):
        if observation is not None:
            self.actor.eval()
            observation = T.tensor(observation, dtype=T.float).to(self.actor.device)
            mu = self.actor(observation).to(self.actor.device)
            noise = T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
            # print("Mu, noise:", mu, noise)
            mu_prime = mu + noise
            self.actor.train()
            return mu_prime.cpu().detach().numpy()
        return np.zeros(self.output_shape)

    def choose_action_test(self, observation):
        if observation is not None:
            self.actor.eval()
            observation = T.tensor(observation, dtype=T.float).to(self.actor.device)
            mu = self.target_actor(observation).to(self.target_actor.device)

            return mu.cpu().detach().numpy()
        return np.zeros(self.output_shape)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self, batch_size=None):
        if batch_size is None:
            if self.memory.mem_cntr < self.batch_size:
                return
            batch_size = self.batch_size

        state, action, reward, new_state, done = self.memory.sample_buffer(batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        done = T.tensor(done).to(self.critic.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.critic.device)
        action = T.tensor(action, dtype=T.float).to(self.critic.device)
        state = T.tensor(state, dtype=T.float).to(self.critic.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        target_actions = self.target_actor.forward(new_state)
        critic_value_ = self.target_critic.forward(new_state, target_actions)
        critic_value = self.critic.forward(state, action)

        target = []
        for j in range(batch_size):
            target.append(reward[j] + self.gamma * critic_value_[j] * done[j])

        target = T.tensor(target).to(self.critic.device)
        target = target.view(batch_size, 1)

        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)

        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def work(self):
        self.target_actor.eval()
        self.target_critic.eval()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()

        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)

        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + \
                                      (1 - tau) * target_critic_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                                     (1 - tau) * target_actor_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()
