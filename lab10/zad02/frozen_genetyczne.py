import gym
import numpy
import pygad

env = gym.make('FrozenLake8x8-v1', is_slippery=False, render_mode="")

observation, info = env.reset(seed=42)


def run_solution(solution):
    reward = 0
    for step in solution:
        observation, reward, terminated, truncated, info = env.step(step)
        if terminated or truncated:
            break
    env.reset(seed=42)
    print(reward)
    return reward

def fitness_func(ga_instance, solution, solution_idx):
    return run_solution(solution)


fitness_function = fitness_func

num_generations = 100
num_parents_mating = 10
sol_per_pop = 50
num_genes = 50
parent_selection_type = "sss"
keep_parents = 10
mutation_percent_genes = 10

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       gene_space=[0,1,2,3],
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       mutation_percent_genes=mutation_percent_genes)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"solution: {solution}")
print(f"fitness value: {solution_fitness}")
env.close()

# przeporwadzenie symulacji dla znalezionego rozwiązania
env = gym.make('FrozenLake8x8-v1', is_slippery=False, render_mode="human")
observation, info = env.reset(seed=42)
print(env.render_mode)
for move in solution:
    action = move
    observation, reward, terminated, truncated, info = env.step(int(action))
    if terminated or truncated:
        break
env.close()

# chromosomy to 50-elementowe (max trasa - 50 kroków) tablice zawierające kolejne ruchy,
# gdzie możliwe ruchy to: 0, 1, 2, 3
# fitness function symuluje rozrgywke dla otrzymanego rozwiazania i zwraca nagrode.
# jesli dojdzie do celu zwraca 1
# jesli nie dojdzie do celu lub wpadnie do dziury zwraca 0
