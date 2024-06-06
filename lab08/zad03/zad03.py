import numpy as np
import pygad
import matplotlib.pyplot as plt
import time

labirynt = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,1,0,0,0,1,0,0,1],
    [1,1,1,0,0,0,1,0,1,1,0,1],
    [1,0,0,0,1,0,1,0,0,0,0,1],
    [1,0,1,0,1,1,0,0,1,1,0,1],
    [1,0,0,1,1,0,0,0,1,0,0,1],
    [1,0,0,0,0,0,1,0,0,0,1,1],
    [1,0,1,0,0,1,1,0,1,0,0,1],
    [1,0,1,1,1,0,0,0,1,1,0,1],
    [1,0,1,0,1,1,0,1,0,1,0,1],
    [1,0,1,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1]
])

# Definicja funkcji fitness
def fitness_func(model, solution, solution_idx):
    x, y = 1, 1  # Początkowe położenie w labiryncie
    penalty = 0
    award = 0

    for i, move in enumerate(solution):
        if x == 10 and y == 10:
            award = 5/(i + 1)  # nagroda za dojście do konća, im mniej kroków (mniejsze i) tym większa nagroda
            break
        elif move == 0 and labirynt[x-1, y] != 1:  # w górę
            x -= 1
        elif move == 1 and labirynt[x+1, y] != 1:  # w dół
            x += 1
        elif move == 2 and labirynt[x, y-1] != 1:  # w lewo
            y -= 1
        elif move == 3 and labirynt[x, y+1] != 1:  # w prawo
            y += 1
        else:
            penalty = (abs(10 - x) + abs(10 - y)) * 0.25 # im blżej początku trasy błąd tym wieksza trasa
            break

    distance_to_exit = abs(10 - x) + abs(10 - y)
    
    if x < 0 or x >= labirynt.shape[0] or y < 0 or y >= labirynt.shape[1]:
        return 0  # Bardzo niska wartość fitness za wyjście poza labirynt
    
    fitness = 1 / (distance_to_exit + 1) - penalty + award

    return fitness

# def on_stop(model, fitness_vals):
#     print(fitness_vals)


num_generations = 5000
sol_per_pop = 40
num_genes = 30
num_parents_mating = 10
keep_parents = 2
gene_space = [0, 1, 2, 3]  #  góra, dół, lewo, prawo
mutation_percent_genes = 7 # 2 zmutowane geny na 30

times = []
for i in range(10):
    print(i, "próba")
    ga_instance = pygad.GA(gene_space=gene_space,
                        num_generations=num_generations,
                        sol_per_pop=sol_per_pop,
                        num_genes=num_genes,
                        mutation_percent_genes=mutation_percent_genes,
                        num_parents_mating=num_parents_mating,
                        keep_parents=keep_parents,
                        fitness_func=fitness_func,
                        stop_criteria="reach_1",
                        #    on_stop=on_stop
                        #    save_best_solutions=True
                        )
    start = time.time()
    ga_instance.run()
    end = time.time()
    times.append(end - start)

print("wszystkie czasy:")
for i in times:
    print(f"{i:.2f} s")
print(f"średni czas: {sum(times)/10:.2f}")


ga_instance = pygad.GA(gene_space=gene_space,
                        num_generations=1000,
                        sol_per_pop=sol_per_pop,
                        num_genes=num_genes,
                        mutation_percent_genes=mutation_percent_genes,
                        num_parents_mating=num_parents_mating,
                        keep_parents=keep_parents,
                        fitness_func=fitness_func,
                        # stop_criteria="reach_1",
                        #    on_stop=on_stop
                        #    save_best_solutions=True
                        )

start = time.time()
ga_instance.run()
end = time.time()


solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Najlepsze rozwiązanie:", solution)
print("Wartość funkcji fitness dla najlepszego rozwiązania:", solution_fitness)
directions_mapping = {
    0: "góra",
    1: "dół",
    2: "lewo",
    3: "prawo"
}

# konwertowanie rozwiązania na słowa
solution_words = [directions_mapping[direction] for direction in solution]

print("Najlepsze rozwiązanie:", solution_words)

def simulate_solution(labirynt, solution):
    x, y = 1, 1
    path = [(x, y)]

    for move in solution:
        if (x, y) == (10, 10):
            print("Sukces! Dotarliśmy do wyjścia.")
            return path
        elif move == 0 and labirynt[x-1, y] != 1:  # w górę
            x -= 1
        elif move == 1 and labirynt[x+1, y] != 1:  # w dół
            x += 1
        elif move == 2 and labirynt[x, y-1] != 1:  # w lewo
            y -= 1
        elif move == 3 and labirynt[x, y+1] != 1:  # w prawo
            y += 1
        else:
            print(f"Nielegalny ruch: ({x}, {y}) przy ruchu {directions_mapping[move]}")
            break  # Nielegalny ruch
        path.append((x, y))
    
    if (x, y) == (10, 10):
        print("Sukces! Dotarliśmy do wyjścia.")
    else:
        print("x,y", x, y)
        print("Nie udało się dotrzeć do wyjścia.")
    return path

# Przeprowadzenie symulacji
path = simulate_solution(labirynt, solution)
print("Ścieżka:", path, ", liczba kroków ", len(path)-1)
ga_instance.plot_fitness()
plt.savefig('fitness_plot.png')
print("=============== czas", end - start)

# dopytać się o stop_criteria, nie zapisuje najlepszego rozwiązania tylko najlepsze rozwiązanie z generacji wcześniejszej

# wszystkie czasy:
# 3.59 s
# 0.27 s
# 0.20 s
# 0.52 s
# 0.37 s
# 0.64 s
# 1.84 s
# 0.30 s
# 0.29 s
# 0.22 s
# średni czas: 0.82