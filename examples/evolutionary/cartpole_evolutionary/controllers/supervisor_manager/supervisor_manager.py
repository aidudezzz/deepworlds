from supervisor_controller import CartpoleSupervisor
import torch.nn as nn

observation_space = 4
action_space = 2

model = nn.Sequential(
    nn.Linear(observation_space, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, action_space),
)

supervisor = CartpoleSupervisor(model=model)
supervisor.train()