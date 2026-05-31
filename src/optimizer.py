import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx   import SBX
from pymoo.operators.mutation.pm     import PM
from pymoo.operators.sampling.rnd    import IntegerRandomSampling
from pymoo.optimize                  import minimize
from pymoo.termination.default       import DefaultMultiObjectiveTermination

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from problem import ShelfOptimizationProblem


def run_optimizer(data: dict,
                  pop_size:    int = 100,
                  n_gen:       int = 200,
                  verbose:     bool = True) -> dict:

    if verbose:
        print("\n" + "="*60)
        print("  NSGA-II Shelf Optimization")
        print("="*60)
        print(f"  Population size : {pop_size}")
        print(f"  Generations     : {n_gen}")
        print(f"  Products        : {len(data['products'])}")
        print(f"  Shelf locations : {len(data['shelves'])}")
        print("="*60)
        print("  Running ... (this takes about 30–60 seconds)")

    problem = ShelfOptimizationProblem(data)

    algorithm = NSGA2(
        pop_size  = pop_size,
        sampling  = IntegerRandomSampling(),
        crossover = SBX(prob=0.9, eta=15, vtype=float, repair=None),
        mutation  = PM(eta=20, vtype=float, repair=None),
        eliminate_duplicates = True,
    )

    termination = DefaultMultiObjectiveTermination(
        xtol=1e-4, cvtol=1e-6, ftol=0.0025,
        period=30, n_max_gen=n_gen
    )

    result = minimize(
        problem,
        algorithm,
        termination,
        seed    = 42,       
        verbose = False,   
    )

    if verbose:
        print(f"  Done! Generations completed : {result.algorithm.n_gen}")
        n_pareto = len(result.X) if result.X is not None else 0
        print(f"  Solutions on Pareto front   : {n_pareto}")

    pareto_solutions = result.X.astype(int)  
    pareto_F         = result.F             

    weights = np.array([
        0.70,  
        0.15,
        0.10,
        0.05,
    ])

    best_idx     = _pick_best(pareto_F, weights)
    best_solution = pareto_solutions[best_idx]

    best_layout   = _decode(best_solution, problem, data)

    improvement   = _compute_improvement(best_layout, data)

    if verbose:
        _print_results(best_layout, improvement)

    return {
        "best_layout"   : best_layout,
        "pareto_front"  : pareto_solutions,
        "pareto_F"      : pareto_F,
        "improvement"   : improvement,
        "problem"       : problem,
    }


def _pick_best(pareto_F: np.ndarray, weights: np.ndarray) -> int:
    return int(np.argmin(pareto_F[:, 0]))


def _decode(solution: np.ndarray, problem, data: dict) -> pd.DataFrame:
    n  = problem.n_products
    products = data["products"]
    shelves  = data["shelves"]

    selected = solution[problem.idx_selected]
    shelf_ids = solution[problem.idx_shelf]
    facings   = solution[problem.idx_facings]

    rows = []
    for p in range(n):
        if selected[p] == 1:
            s = shelf_ids[p]
            rows.append({
                "Product_ID"     : products.loc[p, "Product_ID"],
                "Product_Name"   : products.loc[p, "Product_Name"],
                "Location_ID"    : shelves.loc[s, "Location_ID"],
                "Shelf_Index"    : int(s),
                "Facings"        : int(facings[p]),
                "Current_Facing" : int(products.loc[p, "Current_Facing"]),
                "Unit_Profit"    : float(products.loc[p, "Unit_Profit"]),
                "Monthly_Demand" : int(products.loc[p, "Monthly_Demand"]),
                "Unit_Weight_kg" : float(products.loc[p, "Unit_Weight_kg"]),
                "Visibility_Num" : int(shelves.loc[s, "Visibility_Num"]),
                "Width_cm"       : float(products.loc[p, "Product_Width_cm"]),
                "Total_Revenue"  : float(
                    products.loc[p, "Monthly_Demand"] *
                    products.loc[p, "Unit_Profit"]    *
                    facings[p]
                ),
            })

    df = pd.DataFrame(rows).sort_values("Location_ID").reset_index(drop=True)
    return df


def _compute_improvement(best_layout: pd.DataFrame, data: dict) -> dict:
    products = data["products"]

    current_facing_map = dict(zip(
        products["Product_ID"],
        products["Current_Facing"]
    ))

    current_profit = 0.0
    for _, row in best_layout.iterrows():
        pid         = row["Product_ID"]
        p_row       = products[products["Product_ID"] == pid].iloc[0]
        min_facings = int(p_row["Min_Facing"])
        current_profit += row["Monthly_Demand"] * row["Unit_Profit"] * min_facings

    optimized_profit = best_layout["Total_Revenue"].sum()

    profit_gain_pct = ((optimized_profit - current_profit) / current_profit) * 100 \
                      if current_profit > 0 else 0.0

    return {
        "current_profit"    : current_profit,
        "optimized_profit"  : optimized_profit,
        "profit_gain_pct"   : profit_gain_pct,
        "products_selected" : len(best_layout),
        "total_products"    : len(products),
    }


def _print_results(best_layout: pd.DataFrame, improvement: dict):
    print("\n" + "="*60)
    print("  OPTIMIZATION RESULTS")
    print("="*60)
    print(f"  Products selected   : {improvement['products_selected']} / {improvement['total_products']}")
    print(f"  Current profit      : ₹ {improvement['current_profit']:>12,.0f}")
    print(f"  Optimized profit    : ₹ {improvement['optimized_profit']:>12,.0f}")
    print(f"  Profit improvement  : {improvement['profit_gain_pct']:>+.1f} %")
    print("="*60)
    print("\n  Optimized Layout:")
    print(f"  {'Product':<28} {'Location':<10} {'Facings':>8} {'Revenue':>12}")
    print("  " + "-"*62)
    for _, row in best_layout.iterrows():
        print(f"  {row['Product_Name']:<28} {row['Location_ID']:<10} "
              f"{row['Facings']:>8}  ₹{row['Total_Revenue']:>10,.0f}")
    print()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from data_loader import load_all

    data   = load_all()
    result = run_optimizer(data, pop_size=100, n_gen=200, verbose=True)

    print(f"Pareto front size : {len(result['pareto_front'])} solutions")
    print("Optimizer test complete.\n")
