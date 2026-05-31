import numpy as np
import matplotlib
matplotlib.use("Agg")       
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import pandas as pd

COLORS = [
    "#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0",
    "#00BCD4", "#FF5722", "#8BC34A", "#FFC107", "#607D8B",
]
EMPTY_COLOR  = "#F5F5F5"   
GRID_COLOR   = "#BDBDBD"  


def plot_shelf_layout(best_layout: pd.DataFrame,  n_racks: int = 4,  n_shelves: int = 3, ax=None, save_path: str = None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 6))

    grid_text  = [["" for _ in range(n_racks)] for _ in range(n_shelves)]
    grid_color = [[EMPTY_COLOR] * n_racks for _ in range(n_shelves)]
    grid_facings = [["" for _ in range(n_racks)] for _ in range(n_shelves)]

    product_names = best_layout["Product_Name"].tolist()
    color_map     = {name: COLORS[i % len(COLORS)] for i, name in enumerate(product_names)}

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

    cell_w, cell_h = 1.0, 1.0

    for s in range(n_shelves):
        for r in range(n_racks):
            x = r * cell_w
            y = (n_shelves - 1 - s) * cell_h 

            color = grid_color[s][r]
            rect  = mpatches.FancyBboxPatch(
                (x + 0.05, y + 0.05), cell_w - 0.1, cell_h - 0.1,
                boxstyle="round,pad=0.02",
                facecolor=color, edgecolor="white", linewidth=2
            )
            ax.add_patch(rect)

            if grid_text[s][r]:
                ax.text(x + cell_w/2, y + cell_h*0.60,
                        grid_text[s][r],
                        ha="center", va="center",
                        fontsize=8, fontweight="bold", color="white",
                        wrap=True)
                ax.text(x + cell_w/2, y + cell_h*0.22,
                        grid_facings[s][r],
                        ha="center", va="center",
                        fontsize=7, color="white", alpha=0.9)
            else:
                ax.text(x + cell_w/2, y + cell_h/2,
                        "Empty", ha="center", va="center",
                        fontsize=8, color="#9E9E9E")

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


def plot_pareto_front(pareto_F: np.ndarray, best_idx: int = None, ax=None, save_path: str = None):
    standalone = ax is None 
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5))

    profit     = -pareto_F[:, 0]
    visibility = -pareto_F[:, 1]

    ax.scatter(profit, visibility,
               c="#90CAF9", edgecolors="#1565C0", linewidths=0.8,
               s=60, alpha=0.8, label="Pareto solutions", zorder=2)

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


def plot_before_after(best_layout: pd.DataFrame,
                      data: dict,
                      ax=None,
                      save_path: str = None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 5))

    products_df = data["products"]

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

    bars1 = ax.bar(x - width/2, current_facings,  width, label="Before (min facings)", color="#90CAF9",edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + width/2, optimized_facings, width,
                   label="After (optimized)",   color="#66BB6A",
                   edgecolor="white", linewidth=0.8)

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


def plot_all(result: dict, save: bool = False, output_dir: str = "outputs"):
    import os
    if save:
        os.makedirs(output_dir, exist_ok=True)

    best_layout = result["best_layout"]
    pareto_F    = result["pareto_F"]
    data        = result.get("data")

    best_idx = int(np.argmin(pareto_F[:, 0]))

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Fashion Retail Shelf Optimization — Results",
                 fontsize=16, fontweight="bold", y=0.98)

    ax1 = fig.add_subplot(2, 1, 1)
    plot_shelf_layout(best_layout, ax=ax1)

    ax2 = fig.add_subplot(2, 2, 3)
    plot_pareto_front(pareto_F, best_idx=best_idx, ax=ax2)

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


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from data_loader import load_all
    from optimizer   import run_optimizer

    print("Loading data and running optimizer ...")
    data   = load_all()
    result = run_optimizer(data, pop_size=100, n_gen=200, verbose=True)
    result["data"] = data  

    print("\nGenerating charts ...")
    plot_all(result, save=True, output_dir="outputs")
    print("Done.\n")
