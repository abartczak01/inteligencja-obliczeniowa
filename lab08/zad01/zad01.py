import pygad
import numpy
import matplotlib.pyplot as plt
import time

# items jako lista list [wartość, waga, nazwa]
items = [
    [100, 7, "zegar"],
    [300, 7, "pejzaż"],
    [200, 6, "portret"],
    [40, 2, "laptop"],
    [500, 5, "lampka nocna"],
    [70, 6, "srebrne sztućce"],
    [100, 1, "porcelana"],
    [250, 3, "figura z brązu"],
    [300, 10, "skórzana torebka"],
    [280, 3, "odkurzacz"]
]

# Parametry plecaka
max_weight = 25  # Maksymalna waga plecaka

# Definicja funkcji przystosowania
def fitness_func(model, solution, solution_idx):
    total_value = numpy.sum(numpy.array(solution) * numpy.array([item[0] for item in items]))
    total_weight = numpy.sum(numpy.array(solution) * numpy.array([item[1] for item in items]))
    # jeśli waga jest za duża zwracana jest mała wartość 
    if total_weight > max_weight:
        fitness = 0  # Rozwiązanie przekracza maksymalną wagę
    else:
        fitness = total_value
    return fitness


fitness_function = fitness_func

# Parametry algorytmu genetycznego
sol_per_pop = 10
num_genes = len(items)
num_parents_mating = 5
num_generations = 30
keep_parents = 2
gene_space = [0, 1]
parent_selection_type = "sss"
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 8

# Inicjacja algorytmu genetycznego

# Uruchomienie algorytmu genetycznego
correct_solutions = 0
# ex_time = 0

for _ in range(10):
    ga_instance = pygad.GA(gene_space=gene_space,
                        num_generations=num_generations,
                        num_parents_mating=num_parents_mating,
                        fitness_func=fitness_function,
                        sol_per_pop=sol_per_pop,
                        num_genes=num_genes,
                        parent_selection_type=parent_selection_type,
                        keep_parents=keep_parents,
                        crossover_type=crossover_type,
                        mutation_type=mutation_type,
                        mutation_percent_genes=mutation_percent_genes,
                        stop_criteria="reach_1630",
                        save_best_solutions=True
                        )
    ga_instance.run()

    # Podsumowanie najlepszego rozwiązania
    solution = ga_instance.best_solutions[-1]
    solution_fitness = numpy.sum(numpy.array(solution) * numpy.array([item[0] for item in items]))
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    selected_items = [items[i][-1] for i in range(len(solution)) if solution[i] == 1]
    print("Selected items: ", selected_items)

    total_value = solution_fitness
    print(total_value)
    if total_value == 1630:
        correct_solutions += 1

    print("Total value of the best solution = {total_value}".format(total_value=total_value))

print(f"correct solutions: {correct_solutions/10*100:.2f}%")


correct_solutions = 0
ex_time = 0

while correct_solutions < 10:
    ga_instance = pygad.GA(gene_space=gene_space,
                        num_generations=num_generations,
                        num_parents_mating=num_parents_mating,
                        fitness_func=fitness_function,
                        sol_per_pop=sol_per_pop,
                        num_genes=num_genes,
                        parent_selection_type=parent_selection_type,
                        keep_parents=keep_parents,
                        crossover_type=crossover_type,
                        mutation_type=mutation_type,
                        mutation_percent_genes=mutation_percent_genes,
                        stop_criteria="reach_1630", # kryterium do zatrzymania algorytmu
                        save_best_solutions=True
                        )
    start = time.time()
    ga_instance.run()
    end = time.time()
    solution = ga_instance.best_solutions[-1]    
    total_value = numpy.sum(numpy.array(solution) * numpy.array([item[0] for item in items]))
    print(total_value)
    if total_value == 1630:
        correct_solutions += 1
        ex_time += end - start


print(f"average time for correct solutions: {ex_time/correct_solutions} s")

# ga_instance.plot_fitness()
# plt.savefig('fitness_plot.png')

# average time for correct solutions: 0.013377094268798828 s
# correct solutions: 90.00%
# Selected items:  ['pejzaż', 'portret', 'lampka nocna', 'porcelana', 'figura z brązu', 'odkurzacz']
# solution : [0. 1. 1. 0. 1. 0. 1. 1. 0. 1.]
