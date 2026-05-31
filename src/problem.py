import numpy as np
from pymoo.core.problem import Problem


class ShelfOptimizationProblem(Problem):
    def __init__(self, data: dict):

        self.products     = data["products"]
        self.shelves      = data["shelves"]
        self.cross_matrix = data["cross_matrix"]

        self.n_products = len(self.products)  
        self.n_shelves  = len(self.shelves)  

        self.idx_selected = slice(0,                  self.n_products)
        self.idx_shelf    = slice(self.n_products,    self.n_products * 2)
        self.idx_facings  = slice(self.n_products*2,  self.n_products * 3)

        sel_lb = [0] * self.n_products
        sel_ub = [1] * self.n_products

        shelf_lb = [0]                  * self.n_products
        shelf_ub = [self.n_shelves - 1] * self.n_products

        face_lb = self.products["Min_Facing"].tolist()
        face_ub = self.products["Max_Facing"].tolist()

        lower_bounds = sel_lb + shelf_lb + face_lb
        upper_bounds = sel_ub + shelf_ub + face_ub

        super().__init__(
            n_var        = self.n_products * 3, 
            n_obj        = 4,                  
            n_constr     = 3,                 
            xl           = np.array(lower_bounds, dtype=float),
            xu           = np.array(upper_bounds, dtype=float),
            vtype        = int,              
        )

    def _evaluate(self, X, out, *args, **kwargs):

        pop_size = X.shape[0]

        F = np.zeros((pop_size, 4))  
        G = np.zeros((pop_size, 3)) 

        for i in range(pop_size):
            solution = X[i].astype(int)

            selected = solution[self.idx_selected] 
            shelves  = solution[self.idx_shelf]   
            facings  = solution[self.idx_facings]

            F[i] = self._objectives(selected, shelves, facings)
            G[i] = self._constraints(selected, shelves, facings)

        out["F"] = F
        out["G"] = G

    def _objectives(self, selected, shelves, facings):

        products = self.products

        profit = 0.0
        for p in range(self.n_products):
            if selected[p] == 1:
                profit += (
                    products.loc[p, "Monthly_Demand"] *
                    products.loc[p, "Unit_Profit"]    *
                    facings[p]
                )

        avg_revenue    = profit / max(selected.sum(), 1)
        selection_bonus = 0.05 * avg_revenue * selected.sum()
        profit         += selection_bonus

        visibility = 0.0
        for p in range(self.n_products):
            if selected[p] == 1:
                shelf_idx    = shelves[p]
                shelf_vis    = self.shelves.loc[shelf_idx, "Visibility_Num"]
                product_vis  = products.loc[p, "Visibility_Score"]
                visibility  += shelf_vis * product_vis

        cross_sell = 0.0
        for p1 in range(self.n_products):
            for p2 in range(p1 + 1, self.n_products):
                if selected[p1] == 1 and selected[p2] == 1:
                    pair_score = self.cross_matrix[p1][p2]
                    if pair_score > 0 and shelves[p1] == shelves[p2]:
                        cross_sell += pair_score

        utilization = 0.0
        for s in range(self.n_shelves):
            shelf_length = self.shelves.loc[s, "Shelf Length Limit"]
            used_length  = 0.0
            has_product  = False

            for p in range(self.n_products):
                if selected[p] == 1 and shelves[p] == s:
                    used_length += products.loc[p, "Product_Width_cm"] * facings[p]
                    has_product  = True

            if has_product and shelf_length > 0:
                utilization += min(used_length / shelf_length, 1.0)

        return [-profit, -visibility, -cross_sell, -utilization]

    def _constraints(self, selected, shelves, facings):

        products = self.products

        max_length_violation = 0.0
        for s in range(self.n_shelves):
            shelf_length = self.shelves.loc[s, "Shelf Length Limit"]
            used_length  = 0.0

            for p in range(self.n_products):
                if selected[p] == 1 and shelves[p] == s:
                    used_length += products.loc[p, "Product_Width_cm"] * facings[p]

            violation = used_length - shelf_length
            max_length_violation = max(max_length_violation, violation)

        max_weight_violation = 0.0
        for s in range(self.n_shelves):
            shelf_weight_limit = self.shelves.loc[s, "Shelf Weight Limit"]
            total_weight       = 0.0

            for p in range(self.n_products):
                if selected[p] == 1 and shelves[p] == s:
                    total_weight += (
                        products.loc[p, "Unit_Weight_kg"] *
                        facings[p] *
                        products.loc[p, "Product_stacked"]
                    )

            violation = total_weight - shelf_weight_limit
            max_weight_violation = max(max_weight_violation, violation)

        HEAVY_THRESHOLD = 0.65
        heavy_violation = 0.0

        for p in range(self.n_products):
            if selected[p] == 1:
                weight    = products.loc[p, "Unit_Weight_kg"]
                shelf_idx = shelves[p]
                is_lower  = self.shelves.loc[shelf_idx, "Is_Lower"]

                if weight > HEAVY_THRESHOLD and not is_lower:
                    heavy_violation += 1.0

        return [max_length_violation, max_weight_violation, heavy_violation]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from data_loader import load_all

    print("\nLoading data ...")
    data = load_all()

    print("Creating problem ...")
    problem = ShelfOptimizationProblem(data)

    print(f"\nProblem summary:")
    print(f"  Decision variables : {problem.n_var}  (30 integers)")
    print(f"  Objectives         : {problem.n_obj}  (profit, visibility, cross-sell, utilization)")
    print(f"  Constraints        : {problem.n_constr}  (length, weight, heavy item rule)")
    print(f"  Variable bounds    : lower={problem.xl[:5]} ...  upper={problem.xu[:5]} ...")

    print("\nTesting with a dummy solution (all products on shelf 0, 2 facings each) ...")

    dummy = np.array(
        [1]*10 +   
        [0]*10 + 
        [2]*10, 
        dtype=float
    ).reshape(1, -1)

    out = {}
    problem._evaluate(dummy, out)

    F = out["F"][0]
    G = out["G"][0]

    print(f"\n  Objective scores (raw, negative = better):")
    print(f"    Profit score       : {F[0]:>12.1f}  →  actual profit = {-F[0]:,.0f}")
    print(f"    Visibility score   : {F[1]:>12.1f}  →  actual score  = {-F[1]:.2f}")
    print(f"    Cross-sell score   : {F[2]:>12.1f}  →  actual score  = {-F[2]:.2f}")
    print(f"    Utilization score  : {F[3]:>12.1f}  →  actual score  = {-F[3]:.2f}")

    print(f"\n  Constraint violations (0 = satisfied, >0 = violated):")
    print(f"    Length violation   : {G[0]:.2f}  cm over limit")
    print(f"    Weight violation   : {G[1]:.2f}  kg over limit")
    print(f"    Heavy item rule    : {G[2]:.0f}  products misplaced")

    satisfied = all(g <= 0 for g in G)
    print(f"\n  All constraints satisfied? {'YES' if satisfied else 'NO (expected for dummy solution)'}")
    print("\nProblem formulation working correctly.\n")
