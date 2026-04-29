import numpy as np
from deap import base, creator, tools
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed")

X_train = np.load(os.path.join(DATA_PATH, "X_train.npy"))
y_train = np.load(os.path.join(DATA_PATH, "y_train.npy"))
n_features = X_train.shape[1]

# CONSTANTS
POPULATION_SIZE = 30
MAX_GENERATIONS = 50

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# PSO parameters
c1 = 2.0
c2 = 2.0

# DEAP SETUP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Particle", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# PARTICLE CREATION
def createParticle():
    particle = creator.Particle(np.random.randint(0, 2, size=n_features))
    particle.speed = np.random.uniform(-1, 1, n_features)
    particle.best = particle[:]
    particle.best_fitness = float("inf")
    return particle

toolbox.register("particle", createParticle)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)

# FITNESS FUNCTION

def evaluate(particle):

    selected = np.where(np.array(particle) == 1)[0]

    if len(selected) == 0:
        return (1.0,)

    X_sel = X_train[:, selected]

    model = RandomForestClassifier(n_estimators=50, random_state=RANDOM_SEED)

    acc = cross_val_score(model, X_sel, y_train, cv=3).mean()

    return (1 - acc,)

toolbox.register("evaluate", evaluate)

# UPDATE FUNCTION
def updateParticle(particle, best):

    r1 = np.random.rand(n_features)
    r2 = np.random.rand(n_features)

    cognitive = c1 * r1 * (np.array(particle.best) - np.array(particle))
    social = c2 * r2 * (np.array(best) - np.array(particle))

    particle.speed = particle.speed + cognitive + social

    prob = 1 / (1 + np.exp(-particle.speed))

    particle[:] = np.where(np.random.rand(n_features) < prob, 1, 0)

    if np.sum(particle) == 0:
        particle[np.random.randint(0, n_features)] = 1

toolbox.register("update", updateParticle)

# MAIN LOOP

def run_pso():
    best_fitness_history = []
    population = toolbox.population(n=POPULATION_SIZE)

    best_global = None
    best_global_fitness = float("inf")

    for gen in range(MAX_GENERATIONS):

        for particle in population:

            fitness = toolbox.evaluate(particle)
            particle.fitness = fitness

            # update personal best
            if fitness[0] < particle.best_fitness:
                particle.best = particle[:]
                particle.best_fitness = fitness[0]

            # update global best
            if fitness[0] < best_global_fitness:
                best_global = particle[:]
                best_global_fitness = fitness[0]
        best_fitness_history.append(best_global_fitness)
        # update swarm
        for particle in population:
            toolbox.update(particle, best_global)

        print(f"Generation {gen+1} | Best Fitness: {best_global_fitness:.4f}")
        print(f"num of Selected Features: {len(np.where(np.array(best_global) == 1)[0])}")
        print("Accuracy:", 1 - best_global_fitness)
    selected_features = np.where(np.array(best_global) == 1)[0]

    print("\nBEST FEATURES:", selected_features)
    print("NUM FEATURES:", len(selected_features))
    print("BEST FITNESS:", best_global_fitness)
    print("ACCURACY:", 1 - best_global_fitness)
    print("NUMBER OF SELECTED FEATURES:", len(selected_features))
    # Plot the convergence
    plt.plot(best_fitness_history)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("PSO Feature Selection Convergence")
    plt.show()
    PLOTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")
    os.makedirs(PLOTS_PATH, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_PATH, "pso_convergence.png"))
    return selected_features


if __name__ == "__main__":
    run_pso()