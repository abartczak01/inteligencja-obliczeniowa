import numpy as np
import pygad

# Reprezentacja labiryntu jako dwuwymiarowej macierzy
# labirynt = np.array([
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
#     [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
#     [1, 0, 1, 0, 1, 0, 1, 0, 0, 1],
#     [1, 0, 1, 0, 0, 0, 1, 0, 1, 1],
#     [1, 0, 1, 1, 1, 1, 1, 0, 0, 1],
#     [1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
#     [1, 0, 1, 0, 1, 0, 0, 0, 0, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# ])

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

    for move in solution:
        if move == 0 and labirynt[x-1, y] != 1:  # w górę
            x -= 1
        elif move == 1 and labirynt[x+1, y] != 1:  # w dół
            x += 1
        elif move == 2 and labirynt[x, y-1] != 1:  # w lewo
            y -= 1
        elif move == 3 and labirynt[x, y+1] != 1:  # w prawo
            y += 1
        else:
            penalty += 1  # Ruch na pole ze ścianką lub wyjście poza labirynt

    distance_to_exit = abs(10 - x) + abs(10 - y)
    
    if x < 0 or x >= labirynt.shape[0] or y < 0 or y >= labirynt.shape[1]:
        return 0  # Bardzo niska wartość fitness za wyjście poza labirynt

    fitness = 1 / (distance_to_exit + 1)  # Bliskość do wyjścia
    fitness -= penalty * 0.2  # Kara za ruchy na ścianę lub poza labirynt

    return fitness

# Parametry algorytmu genetycznego
num_generations = 500
sol_per_pop = 50
num_genes = 30
num_parents_mating = 8
gene_space = [0, 1, 2, 3]  #  góra, dół, lewo, prawo
mutation_percent_genes = 0.15

# Inicjacja algorytmu genetycznego
ga_instance = pygad.GA(gene_space=gene_space,
                       num_generations=num_generations,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_func)

# Uruchomienie algorytmu genetycznego
ga_instance.run()

# Wyświetlenie najlepszego rozwiązania
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Najlepsze rozwiązanie:", solution)
print("Wartość funkcji fitness dla najlepszego rozwiązania:", solution_fitness, (1-solution_fitness)/solution_fitness)
directions_mapping = {
    0: "góra",
    1: "dół",
    2: "lewo",
    3: "prawo"
}

# Konwertowanie rozwiązania na słowa
solution_words = [directions_mapping[direction] for direction in solution]

# Wydrukowanie rozwiązania
print("Najlepsze rozwiązanie:", solution_words)

def simulate_solution(labirynt, solution):
    x, y = 1, 1  # Początkowe położenie w labiryncie
    path = [(x, y)]  # Śledzenie ścieżki

    for move in solution:
        if move == 0 and labirynt[x-1, y] != 1:  # w górę
            x -= 1
        elif move == 1 and labirynt[x+1, y] != 1:  # w dół
            x += 1
        elif move == 2 and labirynt[x, y-1] != 1:  # w lewo
            y -= 1
        elif move == 3 and labirynt[x, y+1] != 1:  # w prawo
            y += 1
        else:
            print(f"Nielegalny ruch: ({x}, {y}) przy ruchu {move}")
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
print("Ścieżka:", path)
