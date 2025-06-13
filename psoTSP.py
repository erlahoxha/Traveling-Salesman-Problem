import numpy as np
import random
import matplotlib.pyplot as plt

# Define the TSP problem
cities = [
    (0, 0), (1, 3), (4, 3), (6, 1), (3, 0),
    (5, 5), (2, 6), (8, 8), (6, 4), (7, 1)
]

def calculate_distance(route, distance_matrix):
    """Calculates the total distance of a route given a distance matrix."""
    return sum(distance_matrix[route[i - 1], route[i]] for i in range(len(route)))

def distance(city1, city2):
    return np.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

distance_matrix = np.array([[distance(c1, c2) for c2 in cities] for c1 in cities])

class Particle:
    def __init__(self, route, distance_matrix):
        self.route = route
        self.distance = calculate_distance(route, distance_matrix)
        self.best_route = route
        self.best_distance = self.distance
        self.velocity = []  # Velocity represented as a series of swaps

    def update_position(self):
        for swap in self.velocity:
            i, j = swap
            self.route[i], self.route[j] = self.route[j], self.route[i]

    def evaluate(self, distance_matrix):
        self.distance = calculate_distance(self.route, distance_matrix)
        if self.distance < self.best_distance:
            self.best_distance = self.distance
            self.best_route = self.route.copy()

    def update_velocity(self, global_best_route):
        self.velocity = []
        for _ in range(len(self.route) // 2):
            if random.random() < particle_best_attractor:
                # Swap towards personal best
                i, j = random.sample(range(len(self.route)), 2)
                if self.best_route[i] != self.route[i]:
                    self.velocity.append((i, j))
            else:
                # Swap towards global best
                i, j = random.sample(range(len(self.route)), 2)
                if global_best_route[i] != self.route[i]:
                    self.velocity.append((i, j))

def particle_swarm_optimization(num_cities, num_particles, max_iterations):
    """Implements PSO to solve the TSP."""
    # Initialize particles
    particles = [Particle(random.sample(range(num_cities), num_cities), distance_matrix)
                 for _ in range(num_particles)]

    global_best_route = particles[0].best_route
    global_best_distance = particles[0].best_distance

    for particle in particles:
        if particle.best_distance < global_best_distance:
            global_best_route = particle.best_route
            global_best_distance = particle.best_distance

    for iteration in range(max_iterations):
        for particle in particles:
            particle.update_velocity(global_best_route)
            particle.update_position()
            particle.evaluate(distance_matrix)

            # Update global best
            if particle.best_distance < global_best_distance:
                global_best_route = particle.best_route
                global_best_distance = particle.best_distance

        print(f"Iteration {iteration + 1}/{max_iterations}: Best Distance = {global_best_distance}")

    return global_best_route, global_best_distance

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

# Parameters
num_cities = len(cities)
""" Increasing the particle_best_attractor would drive or attract the individual search more towards each particles' best route 
while decreasing it would lead the search more towards the best solution of an entire swarm (global best route) over multiple iterations.
"""
particle_best_attractor = 0.1
num_particles = 20
max_iterations = 10

best_route, best_distance = particle_swarm_optimization(num_cities, num_particles, max_iterations)
print("Total Cities are: ", cities)
print("Best Route Found:", best_route)
print("Best Route Distance:", best_distance)
plotGraph(best_route)