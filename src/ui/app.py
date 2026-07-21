"""Streamlit UI — Gantt chart + solver interaction (Section 10)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from src.dispatch import solve as dispatch_solve
from src.instance_gen import gen_instance
from src.io_utils import _build_operations, instance_to_dict, solution_to_dict
from src.model import Instance, Solution
from src.solvers.alns import ALNS
from src.solvers.cpsat import CPSAT
from src.solvers.greedy import GreedyERDSPT
from src.validate import validate


def _make_gantt_data(inst: Instance, sol: Solution) -> list[dict]:
    """Build Gantt chart data."""
    rows = []
    for op in inst.ops:
        rows.append({
            "Operation": f"{op.kind[0].upper()}{op.local_id}",
            "Type": op.kind,
            "Start": sol.starts[op.uid],
            "End": sol.starts[op.uid] + op.p,
            "Duration": op.p,
            "Gate": sol.gates[op.uid],
            "Weight": op.w,
            "Cost": op.w * sol.starts[op.uid],
        })
    return rows


def _plot_gantt(inst: Instance, sol: Solution):
    """Render a colorblind-safe Gantt chart using matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    gantt = _make_gantt_data(inst, sol)
    colors = {"delivery": "#0072B2", "pickup": "#E69F00"}

    fig, ax = plt.subplots(figsize=(max(12, inst.T / 2), max(3, inst.G * 0.8)))

    for row in gantt:
        gate_idx = row["Gate"] - 1
        color = colors[row["Type"]]
        rect = mpatches.FancyBboxPatch(
            (row["Start"] - 0.5, gate_idx + 0.1),
            row["Duration"],
            0.8,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
        )
        ax.add_patch(rect)
        ax.text(
            row["Start"] + row["Duration"] / 2,
            gate_idx + 0.5,
            row["Operation"],
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
        )

    for op in inst.ops:
        if op.r > 1:
            gate_idx = sol.gates[op.uid] - 1
            ax.axvline(
                x=op.r - 0.5,
                ymin=gate_idx / inst.G,
                ymax=(gate_idx + 1) / inst.G,
                color="red",
                linewidth=0.5,
                alpha=0.3,
            )

    ax.set_xlim(0.5, inst.T + 0.5)
    ax.set_ylim(-0.1, inst.G + 0.1)
    ax.set_yticks([i + 0.5 for i in range(inst.G)])
    ax.set_yticklabels([f"Gate {g}" for g in range(1, inst.G + 1)])
    ax.set_xlabel("Time Period")
    ax.set_title("Gate Scheduling — Gantt Chart")
    ax.grid(True, alpha=0.3)

    patches = [
        mpatches.Patch(color=colors["delivery"], label="Delivery"),
        mpatches.Patch(color=colors["pickup"], label="Pickup"),
    ]
    ax.legend(handles=patches, loc="upper right")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def main():
    st.set_page_config(page_title="Truck Gate Scheduler", layout="wide")
    st.title("Truck Gate Scheduling — 3-Tier Solver")

    with st.sidebar:
        st.header("Instance Parameters")
        col1, col2 = st.columns(2)
        with col1:
            M = st.number_input("Deliveries (M)", min_value=1, max_value=200, value=5)
        with col2:
            N = st.number_input("Pickups (N)", min_value=1, max_value=200, value=5)

        col3, col4 = st.columns(2)
        with col3:
            G = st.number_input("Gates (G)", min_value=1, max_value=10, value=2)
        with col4:
            T = st.number_input(
                "Horizon (T)",
                min_value=0,
                max_value=500,
                value=0,
                help="0 = auto-size",
            )

        col5, col6 = st.columns(2)
        with col5:
            w1 = st.number_input("Weight w1 (delivery)", min_value=0.0, value=1.0, step=0.1)
        with col6:
            w2 = st.number_input("Weight w2 (pickup)", min_value=0.0, value=1.0, step=0.1)

        seed = st.number_input("Random Seed", min_value=0, value=42)
        time_limit = st.number_input("Time Limit (s)", min_value=0.1, value=5.0, step=0.5)
        algo_label = st.selectbox(
            "Algorithm",
            ["Auto (policy)", "Greedy", "ALNS", "CP-SAT"],
            index=0,
        )
        _algo_to_tier = {
            "Auto (policy)": "auto",
            "Greedy": "greedy",
            "ALNS": "alns",
            "CP-SAT": "cpsat",
        }
        force_tier = _algo_to_tier[algo_label]
        # Side-by-side comparison only applies to Auto; a forced algorithm runs alone.
        if force_tier == "auto":
            compare_all = st.checkbox("Compare all solvers", value=False)
        else:
            compare_all = False

        st.divider()
        uploaded = st.file_uploader("Upload Instance JSON", type=["json"])

    if uploaded is not None:
        data = json.loads(uploaded.read())
        w1_val = float(data.get("w1", 1.0))
        w2_val = float(data.get("w2", 1.0))
        inst = Instance(
            id=data["id"],
            T=data["T"],
            G=data["G"],
            ops=_build_operations(data, w1_val, w2_val),
            w1=w1_val,
            w2=w2_val,
        )
    else:
        inst = gen_instance(
            seed=int(seed),
            M=int(M),
            N=int(N),
            G=int(G),
            T=int(T) if T > 0 else None,
            w1=w1,
            w2=w2,
        )

    st.session_state["inst"] = inst

    st.subheader(f"Instance: {inst.id}  |  K={len(inst.ops)}, G={inst.G}, T={inst.T}")
    st.download_button(
        label="Download instance JSON",
        data=json.dumps(instance_to_dict(inst), indent=2),
        file_name=f"{inst.id}.json",
        mime="application/json",
        key="download_instance",
    )

    if st.button("Solve", type="primary"):
        results: dict[str, Solution] = {}
        tiers: dict[str, str] = {}

        spinner_label = (
            "Running Auto dispatcher..."
            if force_tier == "auto"
            else f"Running {algo_label}..."
        )
        with st.spinner(spinner_label):
            try:
                sol_d, tier = dispatch_solve(
                    inst,
                    seed=int(seed),
                    exact_time_limit=time_limit,
                    alns_time_limit=time_limit,
                    force_tier=force_tier,
                )
                validate(inst, sol_d)
                results["dispatcher"] = sol_d
                tiers["dispatcher"] = tier
                if force_tier == "auto":
                    st.success(f"Dispatcher selected tier: **{tier}**")
                else:
                    st.success(f"Solved with **{tier}**")
            except Exception as e:
                st.error(f"Solve failed: {e}")

        if compare_all and force_tier == "auto":
            with st.spinner("Running Greedy..."):
                g = GreedyERDSPT()
                sol_g = g.solve(inst)
                validate(inst, sol_g)
                results["greedy"] = sol_g

            with st.spinner("Running ALNS..."):
                a = ALNS()
                sol_a = a.solve(inst, time_limit_sec=time_limit, seed=int(seed))
                validate(inst, sol_a)
                results["alns"] = sol_a

            with st.spinner("Running CP-SAT..."):
                try:
                    c = CPSAT()
                    warm = results.get("greedy")
                    sol_c = c.solve(inst, time_limit_sec=time_limit, warm_start=warm)
                    validate(inst, sol_c)
                    results["cpsat"] = sol_c
                except Exception as e:
                    st.warning(f"CP-SAT failed: {e}")

        st.session_state["results"] = results
        st.session_state["tiers"] = tiers
        st.session_state["solved_instance_id"] = inst.id

    results = st.session_state.get("results") or {}
    tiers = st.session_state.get("tiers") or {}
    solved_id = st.session_state.get("solved_instance_id")
    has_current_results = bool(results) and solved_id == inst.id

    if has_current_results:
        st.subheader("Results")
        table_data = []
        for name, sol in results.items():
            row = {
                "Solver": name,
                "Objective": round(sol.objective(inst), 2),
                "Runtime (s)": round(sol.runtime_sec, 3),
                "Proven Optimal": sol.proven_optimal,
                "Tier": tiers.get(name, "—"),
            }
            table_data.append(row)
        st.table(table_data)

        gantt_key = "dispatcher" if "dispatcher" in results else min(
            results, key=lambda n: results[n].objective(inst)
        )
        label = gantt_key
        if gantt_key == "dispatcher":
            label = f"dispatcher ({tiers.get('dispatcher', '?')})"
        st.subheader(f"Gantt Chart — {label}")
        _plot_gantt(inst, results[gantt_key])

        primary = results[gantt_key]
        solver_name = tiers.get(gantt_key, gantt_key)
        sol_payload = solution_to_dict(
            instance_id=inst.id,
            solver=solver_name,
            solution=primary,
            inst=inst,
            proven_optimal=primary.proven_optimal,
            runtime_sec=primary.runtime_sec,
            meta=primary.meta,
        )
        st.download_button(
            label="Download solution JSON",
            data=json.dumps(sol_payload, indent=2),
            file_name=f"{inst.id}_solution.json",
            mime="application/json",
            key="download_solution",
        )
    else:
        st.info("Click 'Solve' to generate a schedule.")


if __name__ == "__main__":
    main()
