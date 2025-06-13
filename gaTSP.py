import random
import numpy as np
import matplotlib.pyplot as plt

# Define the TSP problem
cities = [
    (0, 0), (1, 3), (4, 3), (6, 1), (3, 0),
    (5, 5), (2, 6), (8, 8), (6, 4), (7, 1)
]

def distance(city1, city2):
    return np.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

distance_matrix = np.array([
    [distance(c1, c2) for c2 in cities] for c1 in cities
])

class Individual:
    def __init__(self, route=None):
        if route is None:
            self.route = random.sample(range(len(cities)), len(cities))
        else:
            self.route = route
        self.fitness = self.calculate_fitness()

    def calculate_fitness(self):
        total_distance = sum(
            distance_matrix[self.route[i], self.route[i + 1]]
            for i in range(len(self.route) - 1)
        )
        total_distance += distance_matrix[self.route[-1], self.route[0]]  # Return to start
        return 1 / total_distance

class Population:
    def __init__(self, size):
        self.individuals = [Individual() for _ in range(size)]

    def get_fittest(self):
        return max(self.individuals, key=lambda x: x.fitness)

def selection(population):
    return random.choices(
        population.individuals, 
        weights=[ind.fitness for ind in population.individuals],
        k=2
    )

def crossover(parent1, parent2):
    start, end = sorted(random.sample(range(len(cities)), 2))
    child_route = [None] * len(cities)
    
    child_route[start:end + 1] = parent1.route[start:end + 1]
    pointer = 0
    
    for city in parent2.route:
        if city not in child_route:
            while child_route[pointer] is not None:
                pointer += 1
            child_route[pointer] = city
    
    return Individual(child_route)

def mutate(individual, mutation_rate):
    for swapped in range(len(cities)):
        if random.random() < mutation_rate:
            swap_with = random.randint(0, len(cities) - 1)
            
            city1, city2 = individual.route[swapped], individual.route[swap_with]
            individual.route[swapped], individual.route[swap_with] = city2, city1
    
    individual.fitness = individual.calculate_fitness()

def genetic_algorithm(population_size, generations, mutation_rate):
    population = Population(population_size)
    for generation in range(generations):
        new_population = Population(0)
        
        for _ in range(population_size // 2):
            parent1, parent2 = selection(population)
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            
            mutate(child1, mutation_rate)
            mutate(child2, mutation_rate)
            
            new_population.individuals.append(child1)
            new_population.individuals.append(child2)
        
        population = new_population
        fittest = population.get_fittest()
        print(f"Generation {generation + 1} - Best Fitness: {fittest.fitness}")
    
    return population.get_fittest()

def plotGraph(bestRoute):
    x_val = [cities[x][0] for x in bestRoute]
    x_val.append(cities[bestRoute[0]][0])
    y_val = [cities[x][1] for x in bestRoute]
    y_val.append(cities[bestRoute[0]][1])
    plt.grid(True, color='blue', linewidth=0.1)
    plt.scatter(x_val, y_val)
    plt.plot(x_val, y_val)
    c = 1
    for x in bestRoute:
        if c == 1:
            plt.annotate(c, xy =(cities[x][0], cities[x][1]), xytext =(cities[x][0], cities[x][1]), arrowprops = dict(facecolor = "green", shrink = 0.05),)
        plt.annotate(c, xy =(cities[x][0], cities[x][1]), xytext =(cities[x][0], cities[x][1]))
        c+=1
    plt.show()

# Run the genetic algorithm
best_solution = genetic_algorithm(population_size=500, generations=10, mutation_rate=0.01)
print("Total Cities are: ", cities)
print("Best route found:", best_solution.route)
print("Best route distance:", 1 / best_solution.fitness)
plotGraph(best_solution.route)