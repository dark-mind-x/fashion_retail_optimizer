# =============================================================================
#  app.py
#  PURPOSE : Streamlit web dashboard for the shelf optimization system.
#
#  HOW TO RUN :
#    streamlit run app.py
#
#  WHAT IT DOES :
#    - Lets the user adjust optimizer settings from the sidebar
#    - Runs the NSGA-II optimizer on button click
#    - Shows KPI cards (profit, improvement %, products selected)
#    - Shows the shelf layout, Pareto front, and before/after charts
#    - Lets the user download the results as a CSV file
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from data_loader import load_all
from optimizer   import run_optimizer
from visualizer  import plot_shelf_layout, plot_pareto_front, plot_before_after

st.set_page_config(
    page_title = "Fashion Retail Optimizer",
    page_icon  = "🛍️",
    layout     = "wide",
)

st.markdown("""
<style>
    .kpi-box {
        background: #F8F9FA;
        border-left: 4px solid #2196F3;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
    }
    .kpi-label { font-size: 13px; color: #666; margin-bottom: 4px; }
    .kpi-value { font-size: 26px; font-weight: 700; color: #1A1A2E; }
    .kpi-sub   { font-size: 12px; color: #888; margin-top: 2px; }
    .positive  { color: #2E7D32; }
    .negative  { color: #C62828; }
    .section-title {
        font-size: 17px; font-weight: 600;
        color: #1A1A2E; margin: 1.5rem 0 0.8rem 0;
        border-bottom: 2px solid #E0E0E0; padding-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_data():
    return load_all()


st.title("Fashion Retail Shelf Optimizer")
st.caption("Multi-objective shelf layout optimization using NSGA-II")
st.divider()


with st.sidebar:
    st.header("Optimizer Settings")
    st.caption("Adjust these before running the optimizer.")

    pop_size = st.slider(
        "Population size",
        min_value = 50,
        max_value = 200,
        value     = 100,
        step      = 50,
        help      = "Number of solutions in each generation. Higher = better results but slower."
    )

    n_gen = st.slider(
        "Generations",
        min_value = 50,
        max_value = 300,
        value     = 200,
        step      = 50,
        help      = "How many times the algorithm evolves. Higher = better results but slower."
    )

    st.divider()
    st.caption("**What each setting means:**")
    st.caption("• Population 100, Generations 200 → ~30 seconds")
    st.caption("• Population 200, Generations 300 → ~90 seconds")

    st.divider()
    run_btn = st.button("▶  Run Optimizer", type="primary", use_container_width=True)

    st.divider()
    st.caption("**Project Info**")
    st.caption("Student: Name")


data = get_data()

with st.expander("View raw input data"):
    tab1, tab2, tab3 = st.tabs(["Products", "Shelves", "Cross-Sell Pairs"])
    with tab1:
        st.dataframe(data["products"], use_container_width=True)
    with tab2:
        st.dataframe(data["shelves"], use_container_width=True)
    with tab3:
        st.dataframe(data["cross_sell"], use_container_width=True)


if run_btn:

    with st.spinner("Running NSGA-II optimizer ... this takes about 30–60 seconds."):
        result      = run_optimizer(data, pop_size=pop_size, n_gen=n_gen, verbose=False)
        result["data"] = data
        st.session_state["result"] = result

    st.success("Optimization complete!")


if "result" in st.session_state:
    result      = st.session_state["result"]
    best_layout = result["best_layout"]
    pareto_F    = result["pareto_F"]
    improvement = result["improvement"]
    best_idx    = int(np.argmin(pareto_F[:, 0]))

    st.markdown('<div class="section-title">Key Results</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">Optimized Monthly Profit</div>
            <div class="kpi-value">₹{improvement['optimized_profit']:,.0f}</div>
            <div class="kpi-sub">After optimization</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        gain     = improvement["profit_gain_pct"]
        color_cls = "positive" if gain >= 0 else "negative"
        arrow     = "▲" if gain >= 0 else "▼"
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">Profit Improvement</div>
            <div class="kpi-value {color_cls}">{arrow} {abs(gain):.1f}%</div>
            <div class="kpi-sub">vs minimum facing baseline</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">Products Selected</div>
            <div class="kpi-value">{improvement['products_selected']} / {improvement['total_products']}</div>
            <div class="kpi-sub">placed on shelves</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        pareto_size = len(pareto_F)
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">Pareto Solutions Found</div>
            <div class="kpi-value">{pareto_size}</div>
            <div class="kpi-sub">non-dominated layouts</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="section-title">Optimized Shelf Layout</div>', unsafe_allow_html=True)
    fig1, ax1 = plt.subplots(figsize=(14, 5))
    plot_shelf_layout(best_layout, ax=ax1)
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-title">Pareto Front</div>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        plot_pareto_front(pareto_F, best_idx=best_idx, ax=ax2)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    with col_right:
        st.markdown('<div class="section-title">Before vs After</div>', unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        plot_before_after(best_layout, data, ax=ax3)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    st.divider()

    st.markdown('<div class="section-title">Optimized Layout Detail</div>', unsafe_allow_html=True)

    display_cols = [
        "Product_Name", "Location_ID", "Facings",
        "Current_Facing", "Unit_Profit", "Monthly_Demand", "Total_Revenue"
    ]
    display_df = best_layout[display_cols].copy()
    display_df.columns = [
        "Product", "Location", "Optimized Facings",
        "Current Facings", "Unit Profit (₹)", "Monthly Demand", "Monthly Revenue (₹)"
    ]
    display_df["Monthly Revenue (₹)"] = display_df["Monthly Revenue (₹)"].apply(
        lambda x: f"₹{x:,.0f}"
    )

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown('<div class="section-title">Export Results</div>', unsafe_allow_html=True)

    csv = best_layout.to_csv(index=False).encode("utf-8")
    st.download_button(
        label     = "Download Optimized Layout as CSV",
        data      = csv,
        file_name = "optimized_shelf_layout.csv",
        mime      = "text/csv",
    )

else:
    st.info(" Adjust settings in the sidebar and click **Run Optimizer** to start.")

    st.markdown('<div class="section-title">Products loaded from Excel</div>', unsafe_allow_html=True)
    preview_cols = ["Product_ID", "Product_Name", "Monthly_Demand",
                    "Unit_Profit", "Unit_Weight_kg", "Min_Facing", "Max_Facing"]
    st.dataframe(data["products"][preview_cols], use_container_width=True, hide_index=True)
