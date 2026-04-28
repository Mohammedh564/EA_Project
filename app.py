"""
app.py — Gradio UI for GA Feature Selection
Run: python app.py
Requires: pip install gradio scikit-learn numpy matplotlib
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for Gradio
import matplotlib.pyplot as plt
import gradio as gr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "data", "processed")

# ── Load preprocessed data (produced by preprocessing.py) ──────────────────
X_train = np.load(os.path.join(DATA_PATH, "X_train.npy"))
X_test  = np.load(os.path.join(DATA_PATH, "X_test.npy"))
y_train = np.load(os.path.join(DATA_PATH, "y_train.npy"))
y_test  = np.load(os.path.join(DATA_PATH, "y_test.npy"))
N_FEATURES = X_train.shape[1]


# ═══════════════════════════════════════════════════════════════════════════
# GA CORE — mirrors the notebook exactly, no changes
# ═══════════════════════════════════════════════════════════════════════════

def initialize_population(pop_size, n_features, method="uniform", seed=None):
    if seed is not None:
        np.random.seed(seed)
    if method == "uniform":
        return np.random.randint(0, 2, size=(pop_size, n_features))
    prob_one = 0.2
    return np.random.choice([0, 1], size=(pop_size, n_features),
                             p=[1 - prob_one, prob_one])


def calculate_fitness(chromosome, X, y, config, seed=None):
    if seed is not None:
        np.random.seed(seed)
    selected = np.where(chromosome == 1)[0]
    if len(selected) == 0:
        return -10.0
    clf = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=8,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=42,
    )
    acc = np.mean(cross_val_score(clf, X[:, selected], y,
                                  cv=config["cv_folds"],
                                  scoring="accuracy", n_jobs=-1))
    feature_ratio = len(selected) / len(chromosome)
    base = config["alpha"] * acc - config["beta"] * feature_ratio
    max_allowed = int(len(chromosome) * config["max_features_ratio"])
    penalty = 0.0
    if len(selected) > max_allowed:
        penalty = config["penalty_weight"] * (len(selected) - max_allowed) / len(chromosome)
    return base - penalty


def evaluate_population(population, X, y, config, seed=None):
    return np.array([calculate_fitness(ind, X, y, config, seed)
                     for ind in population])


def roulette_wheel_selection(population, fitness):
    fitness = np.array(fitness)
    fitness = fitness - fitness.min() + 1e-6
    probs   = fitness / fitness.sum()
    return population[np.random.choice(len(population), p=probs)]


def tournament_selection(population, fitness, k=3):
    candidates = np.random.choice(len(population), k, replace=False)
    return population[candidates[np.argmax(fitness[candidates])]]


def single_point_crossover(p1, p2):
    pt = np.random.randint(1, len(p1) - 1)
    return np.concatenate([p1[:pt], p2[pt:]]), np.concatenate([p2[:pt], p1[pt:]])


def uniform_crossover(p1, p2, prob=0.5):
    mask = np.random.rand(len(p1)) < prob
    return np.where(mask, p1, p2), np.where(mask, p2, p1)


def bit_flip_mutation(ind, mutation_rate=0.02):
    ind = ind.copy()
    ind[np.random.rand(len(ind)) < mutation_rate] ^= 1
    if np.sum(ind) == 0:
        ind[np.random.randint(len(ind))] = 1
    return ind


def adaptive_mutation(ind, gen, max_gen, base_rate=0.1, min_rate=0.01):
    ind  = ind.copy()
    rate = max(base_rate * (1 - gen / max_gen), min_rate)
    ind[np.random.rand(len(ind)) < rate] ^= 1
    if np.sum(ind) == 0:
        ind[np.random.randint(len(ind))] = 1
    return ind, rate


def apply_fitness_sharing(fitness, population, sigma=0.5):
    shared = fitness.copy().astype(float)
    for i in range(len(population)):
        niche = sum(
            1.0 - (np.sum(population[i] != population[j]) / len(population[i]) / sigma) ** 2
            for j in range(len(population))
            if np.sum(population[i] != population[j]) / len(population[i]) < sigma
        )
        shared[i] = fitness[i] / max(niche, 1.0)
    return shared


def run_ga(config, X, y, n_features, seed, progress_cb=None):
    """Full GA loop with fitness sharing, elitism, and early stopping."""
    np.random.seed(seed)
    pop_size    = config["pop_size"]
    generations = config["generations"]
    patience    = config["early_stop_patience"]

    population       = initialize_population(pop_size, n_features,
                                              method=config["init_method"], seed=seed)
    best_fitness     = -np.inf
    best_solution    = None
    no_improve       = 0
    acc_history      = []
    feat_history     = []

    for gen in range(generations):
        fitness_vals = evaluate_population(population, X, y, config, seed)
        fitness_vals = apply_fitness_sharing(fitness_vals, population)

        best_idx = np.argmax(fitness_vals)
        if fitness_vals[best_idx] > best_fitness:
            best_fitness  = fitness_vals[best_idx]
            best_solution = population[best_idx].copy()
            no_improve    = 0
        else:
            no_improve += 1

        acc_history.append(fitness_vals[best_idx])
        feat_history.append(int(np.sum(population[best_idx])))

        if progress_cb:
            progress_cb(gen + 1, generations)

        if no_improve >= patience:
            break

        # Elitism: best survives
        new_pop = [population[best_idx].copy()]

        while len(new_pop) < pop_size:
            # Selection
            if config["selection"] == "tournament":
                p1 = tournament_selection(population, fitness_vals)
                p2 = tournament_selection(population, fitness_vals)
            else:
                p1 = roulette_wheel_selection(population, fitness_vals)
                p2 = roulette_wheel_selection(population, fitness_vals)

            # Crossover
            if config["crossover"] == "single":
                c1, c2 = single_point_crossover(p1, p2)
            else:
                c1, c2 = uniform_crossover(p1, p2)

            # Mutation
            if config["mutation"] == "bitflip":
                c1 = bit_flip_mutation(c1)
                c2 = bit_flip_mutation(c2)
            else:
                c1, _ = adaptive_mutation(c1, gen, generations)
                c2, _ = adaptive_mutation(c2, gen, generations)

            new_pop.extend([c1, c2])

        population = np.array(new_pop[:pop_size])

    return best_solution, acc_history, feat_history


# ═══════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ═══════════════════════════════════════════════════════════════════════════

def build_convergence_plot(acc_history, feat_history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    fig.patch.set_facecolor("none")

    ax1.plot(acc_history, color="#1D9E75", linewidth=2, marker="o", markersize=3)
    ax1.set_title("Fitness over generations", fontsize=12)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Best fitness")
    ax1.grid(True, alpha=0.3)

    ax2.plot(feat_history, color="#378ADD", linewidth=2, marker="s", markersize=3)
    ax2.axhline(N_FEATURES, color="gray", linestyle="--", linewidth=1, label="All features")
    ax2.set_title("Feature count over generations", fontsize=12)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Selected features")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def run_ui(selection, crossover, mutation, init_method,
           pop_size, generations, alpha, beta,
           cv_folds, n_estimators, early_stop_patience,
           seed, progress=gr.Progress()):

    config = {
        "selection"           : selection,
        "crossover"           : crossover,
        "mutation"            : mutation,
        "init_method"         : init_method,
        "pop_size"            : int(pop_size),
        "generations"         : int(generations),
        "alpha"               : float(alpha),
        "beta"                : float(beta),
        "cv_folds"            : int(cv_folds),
        "n_estimators"        : int(n_estimators),
        "max_features_ratio"  : 0.6,
        "penalty_weight"      : 2.0,
        "early_stop_patience" : int(early_stop_patience),
    }

    progress(0, desc="Initialising population…")

    def cb(gen, total):
        progress(gen / total, desc=f"Generation {gen} / {total}")

    best_sol, acc_hist, feat_hist = run_ga(
        config, X_train, y_train, N_FEATURES, seed=int(seed),
        progress_cb=cb
    )

    # Final test accuracy (more trees for the real score)
    selected = np.where(best_sol == 1)[0]
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    clf.fit(X_train[:, selected], y_train)
    test_acc = clf.score(X_test[:, selected], y_test)

    n_sel     = int(np.sum(best_sol))
    reduction = round(100 * (1 - n_sel / N_FEATURES), 1)
    gens_run  = len(acc_hist)

    summary = (
        f"**Test accuracy:** {test_acc:.4f}  \n"
        f"**Features selected:** {n_sel} / {N_FEATURES} "
        f"({reduction} % reduction)  \n"
        f"**Generations run:** {gens_run}  \n"
        f"**Selected indices:** {selected.tolist()}"
    )

    fig = build_convergence_plot(acc_hist, feat_hist)
    return summary, fig


# ── Layout ──────────────────────────────────────────────────────────────────
with gr.Blocks(title="GA Feature Selection") as demo:

    gr.Markdown("## 🧬 Genetic Algorithm — Feature Selection\nBreast Cancer Wisconsin dataset · Fitness Sharing · Elitism · Early Stopping")

    with gr.Row():
        # ── Left: operator controls ────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### Operators")
            selection   = gr.Dropdown(["tournament", "roulette"],  value="tournament",  label="Selection")
            crossover   = gr.Dropdown(["single", "uniform"],        value="uniform",     label="Crossover")
            mutation    = gr.Dropdown(["bitflip", "adaptive"],      value="adaptive",    label="Mutation")
            init_method = gr.Dropdown(["uniform", "sparse"],        value="uniform",     label="Initialisation")

            gr.Markdown("### Parameters")
            pop_size     = gr.Slider(10,  100, value=40,  step=5,  label="Population size")
            generations  = gr.Slider(5,   60,  value=30,  step=5,  label="Max generations")
            alpha        = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Alpha (accuracy weight)")
            beta         = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="Beta (feature penalty)")
            cv_folds     = gr.Slider(2,   5,   value=3,   step=1,  label="CV folds")
            n_estimators = gr.Slider(10,  100, value=50,  step=10, label="n_estimators (RF, during GA)")
            patience     = gr.Slider(3,   20,  value=8,   step=1,  label="Early stop patience")
            seed         = gr.Number(value=42, label="Random seed", precision=0)

            run_btn = gr.Button("▶  Run GA", variant="primary")

        # ── Right: results ─────────────────────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### Results")
            summary_box = gr.Markdown("*Results will appear here after running.*")
            plot_box    = gr.Plot(label="Convergence curves")

    run_btn.click(
        fn=run_ui,
        inputs=[selection, crossover, mutation, init_method,
                pop_size, generations, alpha, beta,
                cv_folds, n_estimators, patience, seed],
        outputs=[summary_box, plot_box],
    )

if __name__ == "__main__":
    demo.launch()
