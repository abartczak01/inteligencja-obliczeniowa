import matplotlib.pyplot as plt
import random
from aco import AntColony

plt.style.use("dark_background")

# 20 wierchołków z losowymi parametrami od 0 do 100
COORDS = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(20)]

def plot_nodes(coords, w=12, h=8):
    for x, y in coords:
        plt.plot(x, y, "g.", markersize=15)
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches([w, h])

def plot_all_edges(coords):
    paths = ((a, b) for a in coords for b in coords)
    for a, b in paths:
        plt.plot((a[0], b[0]), (a[1], b[1]))


plot_nodes(COORDS)

# Inicjalizacja algorytmu ACO z różnymi parametrami
colony = AntColony(COORDS, ant_count=300, alpha=0.5, beta=1.2, 
                    pheromone_evaporation_rate=0.40, pheromone_constant=1000.0,
                    iterations=300)

optimal_nodes = colony.get_path()

for i in range(len(optimal_nodes) - 1):
    plt.plot(
        (optimal_nodes[i][0], optimal_nodes[i + 1][0]),
        (optimal_nodes[i][1], optimal_nodes[i + 1][1]),
    )

plt.savefig("zad02.png")

# 1. zmniejszenie ant_count
colony_fewer_ants = AntColony(COORDS, ant_count=100, alpha=0.5, beta=1.2, 
                              pheromone_evaporation_rate=0.40, pheromone_constant=1000.0,
                              iterations=300)
optimal_nodes_fewer_ants = colony_fewer_ants.get_path()

# 2. zwiększenie iterations
colony_more_iterations = AntColony(COORDS, ant_count=300, alpha=0.5, beta=1.2, 
                                   pheromone_evaporation_rate=0.40, pheromone_constant=1000.0,
                                   iterations=1000)
optimal_nodes_more_iterations = colony_more_iterations.get_path()

# 3. zwiekszenie pheromone_evaporation_rate
colony_high_evaporation = AntColony(COORDS, ant_count=300, alpha=0.5, beta=1.2, 
                                    pheromone_evaporation_rate=0.80, pheromone_constant=1000.0,
                                    iterations=300)
optimal_nodes_high_evaporation = colony_high_evaporation.get_path()

# 4. zwiekszenie beta
colony_high_beta = AntColony(COORDS, ant_count=300, alpha=0.5, beta=2.0, 
                             pheromone_evaporation_rate=0.40, pheromone_constant=1000.0,
                             iterations=300)
optimal_nodes_high_beta = colony_high_beta.get_path()

print("Original Parameters:", optimal_nodes)
print("Fewer Ants:", optimal_nodes_fewer_ants)
print("More Iterations:", optimal_nodes_more_iterations)
print("High Evaporation Rate:", optimal_nodes_high_evaporation)
print("High Beta:", optimal_nodes_high_beta)

# Wnioski z eksperymentów:
# 1. Zmniejszenie liczby mrówek może prowadzić do gorszych wyników, ponieważ mniej mrówek oznacza mniej eksploracji przestrzeni poszukiwań.
# 2. Zwiększenie liczby iteracji pozwala algorytmowi na lepsze zbieżności, ale kosztem czasu obliczeń.
# 3. Wyższy współczynnik parowania feromonów może prowadzić do szybszego zbiegania się do lokalnych minimów, ale może również powodować, że algorytm staje się mniej stabilny.
# 4. Zwiększenie wpływu heurystyki (beta) sprawia, że algorytm bardziej polega na informacji heurystycznej niż na feromonach, co może poprawić wyniki w niektórych przypadkach, ale również może powodować nadmierne zaufanie do bieżących informacji heurystycznych.

# Original Parameters: [(100, 28), (97, 26), (88, 30), (95, 44), (70, 45), (71, 35), (57, 55), (43, 57), (27, 74), (11, 73), (10, 71), (5, 50), (29, 14), (62, 0), (73, 2), (95, 12), (94, 74), (91, 96), (67, 87), (37, 95), (100, 28)]
# Fewer Ants: [(95, 44), (88, 30), (97, 26), (100, 28), (95, 12), (73, 2), (62, 0), (29, 14), (5, 50), (10, 71), (11, 73), (27, 74), (37, 95), (67, 87), (91, 96), (94, 74), (70, 45), (71, 35), (57, 55), (43, 57), (95, 44)]
# More Iterations: [(71, 35), (70, 45), (57, 55), (43, 57), (27, 74), (11, 73), (10, 71), (5, 50), (29, 14), (62, 0), (73, 2), (95, 12), (97, 26), (100, 28), (88, 30), (95, 44), (94, 74), (91, 96), (67, 87), (37, 95), (71, 35)]
# High Evaporation Rate: [(71, 35), (70, 45), (57, 55), (43, 57), (27, 74), (11, 73), (10, 71), (5, 50), (29, 14), (62, 0), (73, 2), (95, 12), (97, 26), (100, 28), (88, 30), (95, 44), (94, 74), (91, 96), (67, 87), (37, 95), (71, 35)]
# High Beta: [(57, 55), (43, 57), (27, 74), (11, 73), (10, 71), (5, 50), (29, 14), (62, 0), (73, 2), (95, 12), (97, 26), (100, 28), (88, 30), (95, 44), (70, 45), (71, 35), (94, 74), (91, 96), (67, 87), (37, 95), (57, 55)]
441.682802613807
441.682802613807
441.3194397561528