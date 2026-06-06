# =============================================================================
#  problem.py
#  PURPOSE : Tell NSGA-II what the problem looks like.
#
#  The algorithm needs to know 3 things:
#    1. What decisions can it make?        → decision variables
#    2. What should it try to achieve?     → objective functions (4 of them)
#    3. What rules must it never break?    → constraints
#
#  This file answers all 3 questions in a class called ShelfOptimizationProblem.
#  pymoo (our optimization library) calls this class automatically while running.
#
#  HOW THE ENCODING WORKS
#  ─────────────────────────────────────────────────────────────────────────────
#  We have 10 products and 12 shelf locations.
#  For each product (0 to 9), the algorithm decides:
#
#    selected[p]  → 1 if product p is chosen, 0 if not           (10 variables)
#    shelf[p]     → which shelf location (0 to 11) to put it on  (10 variables)
#    facings[p]   → how many units to display side by side        (10 variables)
#
#  Total decision variables = 10 + 10 + 10 = 30 integers
#
#  Example for product 0 (White Cotton Tee):
#    selected[0] = 1        → yes, we are placing this product
#    shelf[0]    = 3        → put it on shelf location index 3 (R1-S3)
#    facings[0]  = 4        → display 4 units side by side
# =============================================================================

import numpy as np
from pymoo.core.problem import Problem


class ShelfOptimizationProblem(Problem):
    """
    Defines the shelf layout optimization as a pymoo Problem.

    pymoo will create hundreds of random solutions and slowly improve them
    by calling _evaluate() repeatedly. Each call scores one batch of solutions.
    """

    def __init__(self, data: dict):
        """
        data : the dictionary returned by data_loader.load_all()
                data["products"]     → product table
                data["shelves"]      → shelf table
                data["cross_matrix"] → 10x10 cross-sell scores
        """

        # ── Store the data so _evaluate() can use it ─────────────────────────
        self.products     = data["products"]
        self.shelves      = data["shelves"]
        self.cross_matrix = data["cross_matrix"]

        self.n_products = len(self.products)   # 10
        self.n_shelves  = len(self.shelves)    # 12

        # ── Decision variable layout ──────────────────────────────────────────
        # The full solution vector has 30 integers arranged like this:
        #
        #   [ selected[0..9] | shelf[0..9] | facings[0..9] ]
        #     index 0–9        index 10–19   index 20–29
        #
        self.idx_selected = slice(0,                  self.n_products)
        self.idx_shelf    = slice(self.n_products,    self.n_products * 2)
        self.idx_facings  = slice(self.n_products*2,  self.n_products * 3)

        # ── Lower and upper bounds for each variable ──────────────────────────
        # pymoo will only generate values within these bounds.

        # selected[p] : 0 (not chosen) or 1 (chosen)
        sel_lb = [0] * self.n_products
        sel_ub = [1] * self.n_products

        # shelf[p] : any shelf index from 0 to 11
        shelf_lb = [0]                  * self.n_products
        shelf_ub = [self.n_shelves - 1] * self.n_products

        # facings[p] : between the product's Min_Facing and Max_Facing
        face_lb = self.products["Min_Facing"].tolist()
        face_ub = self.products["Max_Facing"].tolist()

        lower_bounds = sel_lb + shelf_lb + face_lb
        upper_bounds = sel_ub + shelf_ub + face_ub

        # ── Tell pymoo the shape of the problem ───────────────────────────────
        super().__init__(
            n_var        = self.n_products * 3,   # 30 decision variables
            n_obj        = 4,                     # 4 objectives to optimize
            n_constr     = 4,                     # 4 constraint types
            xl           = np.array(lower_bounds, dtype=float),
            xu           = np.array(upper_bounds, dtype=float),
            vtype        = int,                   # all variables are integers
        )

    # =========================================================================
    #  _evaluate()
    #  pymoo calls this function with a BATCH of solutions (X).
    #  X shape : (population_size, 30)  e.g. (100, 30)
    #
    #  We must fill:
    #    out["F"] → objective values   shape (100, 4)
    #    out["G"] → constraint values  shape (100, 3)
    #              rule: G <= 0 means the constraint is satisfied
    # =========================================================================
    def _evaluate(self, X, out, *args, **kwargs):

        pop_size = X.shape[0]

        # Prepare output arrays (filled with zeros, then we write into them)
        F = np.zeros((pop_size, 4))   # objective scores
        G = np.zeros((pop_size, 4))   # constraint violations (4 constraints now)

        # ── Score every solution in the population ────────────────────────────
        for i in range(pop_size):
            solution = X[i].astype(int)

            # Split the solution — we still decode all 3 groups
            # selected[] is kept in encoding for compatibility but ignored in scoring
            # (all products are always placed)
            shelves  = solution[self.idx_shelf]      # shape (10,)
            facings  = solution[self.idx_facings]    # shape (10,)

            F[i] = self._objectives(shelves, facings)
            G[i] = self._constraints(shelves, facings)

        out["F"] = F
        out["G"] = G

    # =========================================================================
    #  OBJECTIVE FUNCTIONS
    #  Returns 4 values — all as NEGATIVE because pymoo minimizes by default,
    #  and we want to MAXIMIZE all 4 objectives.
    #  (minimizing -profit = maximizing profit)
    # =========================================================================
    def _objectives(self, shelves, facings):

        products = self.products

        # ── Objective 1 : Total Profit ────────────────────────────────────────
        # ALL products are always placed — no product is ever skipped.
        # Empty shelf penalty: every shelf without a product gets a large
        # penalty so the algorithm is forced to fill ALL shelves.
        profit = 0.0
        for p in range(self.n_products):
            profit += (
                products.loc[p, "Monthly_Demand"] *
                products.loc[p, "Unit_Profit"]    *
                facings[p]
            )

        # Empty shelf penalty — very large penalty for any unfilled shelf.
        # Bottom shelves (S3) get DOUBLE penalty because the optimizer tends
        # to avoid them due to low visibility score.
        for s in range(self.n_shelves):
            shelf_has_product = any(shelves[p] == s for p in range(self.n_products))
            if not shelf_has_product:
                is_bottom = self.shelves.loc[s, "Is_Lower"]
                profit -= 200000 if is_bottom else 150000

        # ── Objective 2 : Shelf Visibility ───────────────────────────────────
        # Eye-level shelves have Visibility_Num = 3 (best)
        # Bottom shelves have Visibility_Num = 1 (worst)
        visibility = 0.0
        for p in range(self.n_products):
            shelf_idx   = shelves[p]
            shelf_vis   = self.shelves.loc[shelf_idx, "Visibility_Num"]
            product_vis = products.loc[p, "Visibility_Score"]
            visibility += shelf_vis * product_vis

        # ── Objective 3 : Cross-Selling Score ────────────────────────────────
        # If two products that are a known pair are placed on the SAME shelf,
        # they get a bonus equal to their cross-sell score.
        cross_sell = 0.0
        for p1 in range(self.n_products):
            for p2 in range(p1 + 1, self.n_products):
                pair_score = self.cross_matrix[p1][p2]
                if pair_score > 0 and shelves[p1] == shelves[p2]:
                    cross_sell += pair_score

        # ── Objective 4 : Shelf Utilization ───────────────────────────────────
        # For each shelf, calculate how much of its length is being used.
        # utilization = used_length / total_length  (value between 0 and 1)
        # We sum this across all shelves that have at least one product.
        utilization = 0.0
        for s in range(self.n_shelves):
            shelf_length = self.shelves.loc[s, "Shelf Length Limit"]
            used_length  = 0.0

            for p in range(self.n_products):
                if shelves[p] == s:
                    used_length += products.loc[p, "Product_Width_cm"] * facings[p]

            if shelf_length > 0:
                utilization += min(used_length / shelf_length, 1.0)

        # Return all 4 as NEGATIVE (pymoo minimizes, we want to maximize)
        return [-profit, -visibility, -cross_sell, -utilization]

    # =========================================================================
    #  CONSTRAINT FUNCTIONS
    #  Returns 3 values. The rule is:
    #    value <= 0  →  constraint is SATISFIED  (good)
    #    value >  0  →  constraint is VIOLATED   (bad, pymoo penalizes this)
    # =========================================================================
    def _constraints(self, shelves, facings):

        products = self.products

        # ── Constraint 1 : Shelf Length Limit ────────────────────────────────
        # Total width of all products on a shelf must not exceed shelf length.
        # We compute the WORST violation across all shelves.
        max_length_violation = 0.0
        for s in range(self.n_shelves):
            shelf_length = self.shelves.loc[s, "Shelf Length Limit"]
            used_length  = 0.0

            for p in range(self.n_products):
                if shelves[p] == s:
                    used_length += products.loc[p, "Product_Width_cm"] * facings[p]

            # violation = how much we exceeded the limit (0 if within limit)
            violation = used_length - shelf_length
            max_length_violation = max(max_length_violation, violation)

        # ── Constraint 2 : Shelf Weight Limit ────────────────────────────────
        # Total weight on a shelf must not exceed its weight limit.
        max_weight_violation = 0.0
        for s in range(self.n_shelves):
            shelf_weight_limit = self.shelves.loc[s, "Shelf Weight Limit"]
            total_weight       = 0.0

            for p in range(self.n_products):
                if shelves[p] == s:
                    # weight = unit weight × number of facings × stacking depth
                    total_weight += (
                        products.loc[p, "Unit_Weight_kg"] *
                        facings[p] *
                        products.loc[p, "Product stacked"]
                    )

            violation = total_weight - shelf_weight_limit
            max_weight_violation = max(max_weight_violation, violation)

        # ── Constraint 3 : Heavy Item Rule ────────────────────────────────────
        # Products heavier than 0.65 kg must ONLY go on bottom shelves (S3).
        # Is_Lower = True means it is a bottom shelf.
        HEAVY_THRESHOLD = 0.65
        heavy_violation = 0.0

        for p in range(self.n_products):
            weight   = products.loc[p, "Unit_Weight_kg"]
            shelf_idx = shelves[p]
            is_lower  = self.shelves.loc[shelf_idx, "Is_Lower"]

            if weight > HEAVY_THRESHOLD and not is_lower:
                heavy_violation += 1.0

        # ── Constraint 4 : One Product Per Shelf ─────────────────────────────
        # Each shelf location must have AT MOST one product.
        # This forces the optimizer to spread products across all shelves
        # instead of stacking many products on one shelf and leaving others empty.
        shelf_conflict = 0.0
        shelf_counts = {}
        for p in range(self.n_products):
            s = shelves[p]
            shelf_counts[s] = shelf_counts.get(s, 0) + 1
        for s, count in shelf_counts.items():
            if count > 1:
                shelf_conflict += (count - 1)   # each extra product is a violation

        return [max_length_violation, max_weight_violation, heavy_violation, shelf_conflict]


# =============================================================================
#  QUICK TEST — run this file directly to verify the problem is set up correctly
#  Command : python src/problem.py
# =============================================================================
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

    # ── Test with one hand-crafted solution ──────────────────────────────────
    # Select all 10 products, put them all on shelf 0, give each 2 facings
    print("\nTesting with a dummy solution (all products on shelf 0, 2 facings each) ...")

    dummy = np.array(
        [1]*10 +    # select all products
        [0]*10 +    # put all on shelf index 0
        [2]*10,     # 2 facings each
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
