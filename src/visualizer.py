# =============================================================================
#  visualizer.py
#  PURPOSE : Create 3 charts from the optimizer results.
#
#  CHART 1 — Shelf Layout Heatmap
#             A 4×3 grid (4 racks, 3 shelves each) showing which product
#             is placed on which shelf location.
#
#  CHART 2 — Pareto Front Plot
#             A scatter plot showing the trade-off between profit and
#             visibility across all 100 Pareto-optimal solutions.
#             The chosen best solution is highlighted.
#
#  CHART 3 — Before vs After Comparison
#             A grouped bar chart comparing current facings vs optimized
#             facings for each selected product.
#
#  Usage:
#    from visualizer import plot_all
#    plot_all(result)           # shows all 3 charts
#    plot_all(result, save=True) # also saves PNG files
# =============================================================================

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for all systems)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import pandas as pd

# Consistent color palette across all charts
COLORS = [
    "#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0",
    "#00BCD4", "#FF5722", "#8BC34A", "#FFC107", "#607D8B",
]
EMPTY_COLOR  = "#F5F5F5"   # grey for empty shelf cells
GRID_COLOR   = "#BDBDBD"   # shelf grid lines


# =============================================================================
#  CHART 1 : Shelf Layout Heatmap
#  Shows the 4×3 rack-shelf grid with product names inside each cell.
# =============================================================================
def plot_shelf_layout(best_layout: pd.DataFrame,
                      n_racks: int = 4,
                      n_shelves: int = 3,
                      ax=None,
                      save_path: str = None,
                      cross_sell: pd.DataFrame = None):
    """
    Draws the shelf layout as a coloured grid.
    Cross-sell pairs on the same shelf get a gold border indicator.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 6))

    grid_text    = [["" for _ in range(n_racks)] for _ in range(n_shelves)]
    grid_color   = [[EMPTY_COLOR] * n_racks for _ in range(n_shelves)]
    grid_facings = [["" for _ in range(n_racks)] for _ in range(n_shelves)]
    grid_cross   = [[False] * n_racks for _ in range(n_shelves)]

    product_names = best_layout["Product_Name"].tolist()
    color_map     = {name: COLORS[i % len(COLORS)] for i, name in enumerate(product_names)}

    # Build location → product name map
    loc_to_name = dict(zip(best_layout["Location_ID"], best_layout["Product_Name"]))

    # Find cross-sell pairs that ended up on the same shelf
    cross_locations = set()
    if cross_sell is not None:
        for _, row in cross_sell.iterrows():
            p1, p2 = row["Product_1"], row["Product_2"]
            loc1 = best_layout[best_layout["Product_Name"] == p1]["Location_ID"].values
            loc2 = best_layout[best_layout["Product_Name"] == p2]["Location_ID"].values
            if len(loc1) and len(loc2) and loc1[0] == loc2[0]:
                cross_locations.add(loc1[0])

    for _, row in best_layout.iterrows():
        loc    = row["Location_ID"]
        rack   = int(loc[1]) - 1
        shelf  = int(loc[4]) - 1
        name   = row["Product_Name"]
        facings = int(row["Facings"])

        words = name.split()
        if len(words) > 2:
            label = " ".join(words[:2]) + "\n" + " ".join(words[2:])
        else:
            label = name

        grid_text[shelf][rack]    = label
        grid_color[shelf][rack]   = color_map[name]
        grid_facings[shelf][rack] = f"{facings} facings"
        grid_cross[shelf][rack]   = loc in cross_locations

    cell_w, cell_h = 1.0, 1.0

    for s in range(n_shelves):
        for r in range(n_racks):
            x = r * cell_w
            y = (n_shelves - 1 - s) * cell_h

            color     = grid_color[s][r]
            is_cross  = grid_cross[s][r]

            # Gold thick border for cross-sell pairs, white for normal
            edge_color  = "#FFD700" if is_cross else "white"
            edge_width  = 4 if is_cross else 2

            rect = mpatches.FancyBboxPatch(
                (x + 0.05, y + 0.05), cell_w - 0.1, cell_h - 0.1,
                boxstyle="round,pad=0.02",
                facecolor=color, edgecolor=edge_color, linewidth=edge_width
            )
            ax.add_patch(rect)

            if grid_text[s][r]:
                ax.text(x + cell_w/2, y + cell_h*0.62,
                        grid_text[s][r],
                        ha="center", va="center",
                        fontsize=8, fontweight="bold", color="white")
                ax.text(x + cell_w/2, y + cell_h*0.25,
                        grid_facings[s][r],
                        ha="center", va="center",
                        fontsize=7, color="white", alpha=0.9)
                # Small cross-sell badge
                if is_cross:
                    ax.text(x + cell_w - 0.12, y + cell_h - 0.12,
                            "↔", ha="center", va="center",
                            fontsize=10, color="#FFD700", fontweight="bold")
            else:
                ax.text(x + cell_w/2, y + cell_h/2,
                        "Empty", ha="center", va="center",
                        fontsize=8, color="#9E9E9E")

    # ── Axis labels ───────────────────────────────────────────────────────────
    ax.set_xlim(0, n_racks)
    ax.set_ylim(0, n_shelves)
    ax.set_xticks([i + 0.5 for i in range(n_racks)])
    ax.set_xticklabels([f"Rack {i+1}" for i in range(n_racks)], fontsize=10)
    ax.set_yticks([i + 0.5 for i in range(n_shelves)])
    ax.set_yticklabels(["Bottom (S3)", "Mid (S2)", "Eye-Level (S1)"],
                       fontsize=10)
    ax.set_title("Optimized Shelf Layout", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {save_path}")
        plt.show()
        plt.close()


# =============================================================================
#  CHART 2 : Pareto Front Plot
#  Shows profit vs visibility trade-off for all 100 Pareto solutions.
#  The best chosen solution is highlighted with a star.
# =============================================================================
def plot_pareto_front(pareto_F: np.ndarray,
                      best_idx: int = None,
                      ax=None,
                      save_path: str = None):
    """
    Scatter plot of all Pareto-optimal solutions.
    X-axis = profit, Y-axis = visibility score.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Convert back from negative (pymoo stores minimized values)
    profit     = -pareto_F[:, 0]
    visibility = -pareto_F[:, 1]

    # All Pareto solutions
    ax.scatter(profit, visibility,
               c="#90CAF9", edgecolors="#1565C0", linewidths=0.8,
               s=60, alpha=0.8, label="Pareto solutions", zorder=2)

    # Highlight the chosen best solution
    if best_idx is not None:
        ax.scatter(profit[best_idx], visibility[best_idx],
                   c="#F44336", edgecolors="#B71C1C", linewidths=1.5,
                   s=200, marker="*", label="Selected solution", zorder=3)
        ax.annotate("  Best",
                    (profit[best_idx], visibility[best_idx]),
                    fontsize=9, color="#B71C1C", fontweight="bold")

    ax.set_xlabel("Total Profit  (₹)", fontsize=10)
    ax.set_ylabel("Visibility Score", fontsize=10)
    ax.set_title("Pareto Front  —  Profit vs Visibility Trade-off",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"₹{x/1e5:.1f}L")
    )

    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {save_path}")
        plt.show()
        plt.close()


# =============================================================================
#  CHART 3 : Before vs After Comparison Bar Chart
#  Grouped bars: current (min) facings vs optimized facings per product.
# =============================================================================
def plot_before_after(best_layout: pd.DataFrame,
                      data: dict,
                      ax=None,
                      save_path: str = None):
    """
    Grouped bar chart showing current facings vs optimized facings
    for each selected product.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 5))

    products_df = data["products"]

    # Build comparison data
    names            = []
    current_facings  = []
    optimized_facings = []
    revenues_current  = []
    revenues_opt      = []

    for _, row in best_layout.iterrows():
        pid     = row["Product_ID"]
        p_row   = products_df[products_df["Product_ID"] == pid].iloc[0]

        names.append(row["Product_Name"].replace(" ", "\n"))
        current_facings.append(int(p_row["Min_Facing"]))
        optimized_facings.append(int(row["Facings"]))
        revenues_current.append(
            p_row["Monthly_Demand"] * p_row["Unit_Profit"] * p_row["Min_Facing"]
        )
        revenues_opt.append(row["Total_Revenue"])

    x      = np.arange(len(names))
    width  = 0.35

    bars1 = ax.bar(x - width/2, current_facings,  width,
                   label="Before (min facings)", color="#90CAF9",
                   edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + width/2, optimized_facings, width,
                   label="After (optimized)",   color="#66BB6A",
                   edgecolor="white", linewidth=0.8)

    # Revenue labels on top of each bar
    for bar, rev in zip(bars1, revenues_current):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.05,
                f"₹{rev/1000:.0f}K",
                ha="center", va="bottom", fontsize=7, color="#1565C0")

    for bar, rev in zip(bars2, revenues_opt):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.05,
                f"₹{rev/1000:.0f}K",
                ha="center", va="bottom", fontsize=7, color="#2E7D32")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Number of Facings", fontsize=10)
    ax.set_title("Before vs After  —  Facings and Revenue per Product",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(optimized_facings) + 2)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {save_path}")
        plt.show()
        plt.close()


# =============================================================================
#  MAIN FUNCTION — plot all 3 charts together in one figure
# =============================================================================
def plot_all(result: dict, save: bool = False, output_dir: str = "outputs"):
    """
    Renders all 3 charts in a single figure.

    Parameters
    ----------
    result     : dict returned by run_optimizer()
    save       : if True, saves PNG files to output_dir
    output_dir : folder to save images (created if missing)
    """
    import os
    if save:
        os.makedirs(output_dir, exist_ok=True)

    best_layout = result["best_layout"]
    pareto_F    = result["pareto_F"]
    data        = result.get("data")

    # Find the index of the best solution in the Pareto front
    best_idx = int(np.argmin(pareto_F[:, 0]))

    # ── Figure layout: 3 subplots ─────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Fashion Retail Shelf Optimization — Results",
                 fontsize=16, fontweight="bold", y=0.98)

    # Chart 1 : shelf layout (top, full width)
    ax1 = fig.add_subplot(2, 1, 1)
    plot_shelf_layout(best_layout, ax=ax1)

    # Chart 2 : Pareto front (bottom left)
    ax2 = fig.add_subplot(2, 2, 3)
    plot_pareto_front(pareto_F, best_idx=best_idx, ax=ax2)

    # Chart 3 : before vs after (bottom right)
    ax3 = fig.add_subplot(2, 2, 4)
    if data:
        plot_before_after(best_layout, data, ax=ax3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        path = os.path.join(output_dir, "optimization_results.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"\n  Saved combined chart → {path}")

    plt.close()
    print("  Charts generated successfully.")


# =============================================================================
#  RUN DIRECTLY TO TEST
#  Command : python src/visualizer.py
# =============================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from data_loader import load_all
    from optimizer   import run_optimizer

    print("Loading data and running optimizer ...")
    data   = load_all()
    result = run_optimizer(data, pop_size=100, n_gen=200, verbose=True)
    result["data"] = data   # attach data so plot_before_after can use it

    print("\nGenerating charts ...")
    plot_all(result, save=True, output_dir="outputs")
    print("Done.\n")
