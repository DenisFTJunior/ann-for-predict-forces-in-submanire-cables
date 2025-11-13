"""Optimization utilities for ANN models: random search, evolutionary (GA), and Keras Tuner.

Each function takes a `build_fn(input_dim, hidden_layers, activation, dropout, learning_rate)`
that returns a compiled tf.keras.Model, and dataset tuples (x_train, y_train, x_val, y_val).
They return: (best_params: dict, best_model: tf.keras.Model)
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple, List, Optional
import random
import numpy as np


def _generate_hidden_layer_options(
    input_dim: int,
    *,
    min_layers: int = 2,
    max_layers: int = 6,
    min_units: int = 16,
    max_units: int = 256,
    step: Optional[int] = None,
) -> List[tuple]:
    """Generate tuples of layer sizes dynamically based on input_dim and bounds.

    - min_layers/max_layers: number of hidden layers to include
    - min_units/max_units: units per layer (will be clamped to [min_units, max_units])
    - step: granularity of units
    Strategy: sample unit options around input_dim with bounds.
    """
    # Ensure sensible bounds
    min_layers = max(1, min_layers)
    max_layers = max(min_layers, max_layers)
    min_units = max(4, min_units)
    max_units = max(min_units, max_units)
    if not step or step <= 0:
        # Default step scaled to range
        span = max_units - min_units
        step = max(8, span // 8) if span > 0 else 16

    # Build candidate unit values around input_dim
    base = max(min_units, min(max_units, int(round(input_dim))))
    candidates = set()
    # Sweep around base using step multiples
    for k in range(-4, 5):
        val = base + k * step
        if min_units <= val <= max_units:
            candidates.add(val)
    # Always include boundaries
    candidates.update({min_units, max_units})
    unit_values = sorted(candidates)

    options: List[tuple] = []
    for L in range(min_layers, max_layers + 1):
        # Create a few patterns: descending, ascending, flat
        if unit_values:
            # flat
            options.append(tuple([base] * L))
            # descending
            desc = [max(min_units, min(max_units, base - i * step)) for i in range(L)]
            options.append(tuple(desc))
            # ascending
            asc = [max(min_units, min(max_units, base + i * step)) for i in range(L)]
            options.append(tuple(asc))
        else:
            options.append(tuple([base] * L))
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for opt in options:
        if opt not in seen:
            uniq.append(opt)
            seen.add(opt)
    return uniq


def random_search_ann(
    build_fn: Callable,
    x_train,
    y_train,
    x_val,
    y_val,
    *,
    trials: int = 20,
    epochs: int = 50,
    batch_size: int = 32,
    min_layers: int = 2,
    max_layers: int = 6,
    min_units: int = 16,
    max_units: int = 256,
    step: int = 16,
) -> Tuple[Dict, object]:
    """Simple random search over a small hyperparameter space."""
    input_dim = x_train.shape[-1]
    hidden_choices = _generate_hidden_layer_options(
        input_dim,
        min_layers=min_layers,
        max_layers=max_layers,
        min_units=min_units,
        max_units=max_units,
        step=step,
    )
    activations = ["relu", "tanh", "elu", "sigmoid", "softmax"]
    dropouts = [0.0, 0.1, 0.2, 0.3]
    lrs = [1e-2, 5e-3, 1e-3, 5e-4]

    best = (float("inf"), {})  # (val_loss, params)
    best_model = None
    input_dim = x_train.shape[-1]

    for _ in range(trials):
        params = {
            "hidden_layers": random.choice(hidden_choices),
            "activation": random.choice(activations),
            "dropout": random.choice(dropouts),
            "learning_rate": random.choice(lrs),
        }
        model = build_fn(input_dim, **params)
        hist = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                         validation_data=(x_val, y_val), verbose=0)
        # or val_loss or infinity if missing
        val_loss = float(min(hist.history.get("val_loss", [np.inf])))
        if val_loss < best[0]:
            best = (val_loss, params)
            best_model = model
    return best[1], best_model


def evolutionary_optimize_ann(
    build_fn: Callable,
    x_train,
    y_train,
    x_val,
    y_val,
    *,
    population: int = 4,
    generations: int = 2,
    epochs: int = 30,
    batch_size: int = 32,
    min_layers: int = 2,
    max_layers: int = 6,
    min_units: int = 16,
    max_units: int = 256,
) -> Tuple[Dict, object]:
    
    input_dim = x_train.shape[-1]

    hidden_pool = _generate_hidden_layer_options(
        input_dim,
        min_layers=min_layers,
        max_layers=max_layers,
        min_units=min_units,
        max_units=max_units,
    )

    def sample_individual():
        return {
            "hidden_layers": random.choice(hidden_pool) if hidden_pool else (input_dim,),
            "activation": random.choice(["relu", "tanh"]),
            "dropout": random.choice([0.0, 0.1, 0.2, 0.3]),
            "learning_rate": random.choice([1e-2, 5e-3, 1e-3, 5e-4]),
        }

    def mutate(p):
        q = dict(p)
        key = random.choice(list(q.keys()))
        if key == "hidden_layers":
            q[key] = random.choice(hidden_pool) if hidden_pool else q[key]
        elif key == "activation":
            q[key] = random.choice(["relu", "tanh"])
        elif key == "dropout":
            q[key] = max(0.0, min(0.5, q[key] + random.choice([-0.1, 0.0, 0.1])))
        elif key == "learning_rate":
            q[key] = random.choice([1e-2, 5e-3, 1e-3, 5e-4])
        return q

    def crossover(a, b):
        child = {}
        for k in a:
            child[k] = random.choice([a[k], b[k]])
        return child

    def evaluate(params):
        model = build_fn(input_dim, **params)
        hist = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                         validation_data=(x_val, y_val), verbose=0)
        return float(min(hist.history.get("val_loss", [np.inf]))), model

    def init_population():
        return [sample_individual() for _ in range(population)]

    def score_population(pop):
        return sorted((evaluate(ind) + (ind,)) for ind in pop)

    def select_survivors(scored):
        return scored[: max(2, len(scored)//2)]

    def make_children(parents, target_count):
        result = []
        parent_params = [p for _, _, p in parents]
        while len(result) < target_count:
            a, b = random.sample(parent_params, 2)
            result.append(mutate(crossover(a, b)))
        return result

    # initialize and iterate generations
    scores = score_population(init_population())
    for _ in range(generations):
        print(f"Generation best val_loss: {scores[0][0]:.4f}")
        survivors = select_survivors(scores)
        need = max(0, population - len(survivors))
        children = make_children(survivors, need)
        scores = score_population([p for _, _, p in survivors] + children)

    _, best_model, best_params= scores[0]
    print(f"Best params after evolution: {best_params}")
    print(f"val_loss: {scores[0]}")
    return best_params, best_model


def keras_tuner_optimize_ann(
    build_fn: Callable,
    x_train,
    y_train,
    x_val,
    y_val,
    *,
    max_trials: int = 15,
    executions_per_trial: int = 1,
    epochs: int = 50,
    batch_size: int = 32,
) -> Tuple[Dict, object]:
    """Keras Tuner BayesianOptimization over a small space."""
    import importlib
    kt = importlib.import_module('keras_tuner')  # raises ImportError if missing

    tuner = kt.BayesianOptimization(
        build_fn,
        objective="val_loss",
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory="kt_logs",
        project_name="ann_opt",
    )

    tuner.search(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size, verbose=0)
    best_hp = tuner.get_best_hyperparameters(1)[0]
    params = {
        "num_layers": best_hp.get("num_layers"),
        "activation": best_hp.get("activation"),
        "dropout": best_hp.get("dropout"),
        "learning_rate": best_hp.get("learning_rate"),
    }
    best_model = tuner.get_best_models(1)[0]
    return params, best_model
