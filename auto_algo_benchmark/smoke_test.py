from benchmark_harness.tasks import builtin_task_specs, evaluate_solver_callable

def solve(objective, budget, dim, lower_bound, upper_bound, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    best_x = rng.uniform(lower_bound, upper_bound, size=dim)
    best_f = float(objective(best_x))
    history = [best_f]
    for _ in range(1, budget):
        x = rng.uniform(lower_bound, upper_bound, size=dim)
        fx = float(objective(x))
        if fx < best_f:
            best_x = x
            best_f = fx
        history.append(best_f)
    return {"best_x": best_x, "best_f": best_f, "history": history}

if __name__ == "__main__":
    task = builtin_task_specs({"budget": 40, "eval_seeds": [1, 2, 3]})["mixed_5d"]
    result = evaluate_solver_callable(solve, task)
    print(result)


# 