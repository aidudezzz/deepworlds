from supervisor_controller import CartpoleSupervisor
import torch.nn as nn
import matplotlib.pyplot as plt
import wandb

observation_space = 4
action_space = 2

model = nn.Sequential(
    nn.Linear(observation_space, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, action_space),
)

wandb.init(project="deepbots-evo")

supervisor1 = CartpoleSupervisor(model=model)
_, _, _, fitness1 = supervisor1.train(
                                    num_generations=150, 
                                    num_parents_mating=5,
                                    num_solutions=10,
                                    parent_selection_type="sss", 
                                    crossover_type="single_point",
                                    mutation_type="random",
                                    mutation_percent_genes=10, 
                                    keep_parents=-1,
                                    )

'''supervisor2 = CartpoleSupervisor(model=model)
_, _, _, fitness2 = supervisor2.train(
                                    num_generations=75, 
                                    num_parents_mating=8,
                                    num_solutions=10,
                                    parent_selection_type="sss", 
                                    crossover_type="single_point",
                                    mutation_type="random",
                                    mutation_percent_genes=10, 
                                    keep_parents=-1,
                                    )

supervisor3 = CartpoleSupervisor(model=model)
_, _, _, fitness3 = supervisor3.train(
                                    num_generations=75, 
                                    num_parents_mating=5,
                                    num_solutions=20,
                                    parent_selection_type="sss", 
                                    crossover_type="single_point",
                                    mutation_type="random",
                                    mutation_percent_genes=10, 
                                    keep_parents=-1,
                                    )

supervisor4 = CartpoleSupervisor(model=model)
_, _, _, fitness4 = supervisor4.train(
                                    num_generations=75, 
                                    num_parents_mating=5,
                                    num_solutions=10,
                                    parent_selection_type="tournament", 
                                    crossover_type="single_point",
                                    mutation_type="random",
                                    mutation_percent_genes=10, 
                                    keep_parents=-1,
                                    K_tournament=1,
                                    )

supervisor5 = CartpoleSupervisor(model=model)
_, _, _, fitness5 = supervisor5.train(
                                    num_generations=75, 
                                    num_parents_mating=5,
                                    num_solutions=10,
                                    parent_selection_type="sus", 
                                    crossover_type="single_point",
                                    mutation_type="random",
                                    mutation_percent_genes=10, 
                                    keep_parents=-1,
                                    )'''

