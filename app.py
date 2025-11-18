# app.py — Streamlit UI for Phase 8 results (Pareto, heatmap, radar, flat results)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import os
from pathlib import Path
import io

st.set_page_config(page_title="Polypharmacy MORL — Phase 8 Explorer", layout="wide")

# --------------------
# CONFIG: update paths here if needed
# --------------------
BASE_DIR = Path(".")
# typical outputs from your Phase8 summary JSON
PHASE8_DIR = BASE_DIR / "phase8_results"  # default folder; replace if your folder is different
# fallback to path you posted earlier (change if needed)
PHASE8_DIR = Path("/content/drive/MyDrive/ML Patent/polypharmacy_project/phase7_results/phase8_results_20251118_050009") if Path("/content/drive").exists() else PHASE8_DIR

FILES = {
    "flat_csv": PHASE8_DIR / "phase8_flat_results.csv",
    "pareto_csv": PHASE8_DIR / "pareto_front_mean_by_weight.csv",
    "pareto_plot": PHASE8_DIR / "pareto_front_plot.png",
    "heatmap": PHASE8_DIR / "action_frequency_heatmap.png",
    "radar": PHASE8_DIR / "tradeoff_radar_plots.png",
    "corr": PHASE8_DIR / "reward_correlation_matrix.png",
    "raw_json": PHASE8_DIR / "phase8_raw_results.json",
}

# --------------------
# Utilities
# --------------------
def file_exists(p: Path):
    return p.exists() and p.is_file()

def try_load_csv(p: Path):
    if file_exists(p):
        try:
            return pd.read_csv(p)
        except Exception as e:
            st.error(f"Failed to read CSV {p}: {e}")
            return None
    return None

def display_image(p: Path, width=700):
    if file_exists(p):
        st.image(str(p), use_column_width=True)
    else:
        st.info(f"Image not found: {p.name}")

# --------------------
# Action mapping (simple clinical interpretation)
# --------------------
# Replace or extend this mapping with your real action->drug combination table.
ACTION_MAP = {
    0: "Action 0 — Reduce-risk policy (prefer safer combos / deprescribe candidate)",
    1: "Action 1 — Maintain current regimen (conservative)",
    # if you ever expand action space, fill here.
}

# If you have a more detailed action->drug mapping CSV (index -> drug set),
# point ACTION_DRUG_MAP_CSV to it and app will load.
ACTION_DRUG_MAP_CSV = BASE_DIR / "action_to_drug_map.csv"  # optional

# --------------------
# Sidebar controls
# --------------------
st.sidebar.title("Phase 8 — Explorer")
st.sidebar.markdown("Choose visualization and dataset")
viz = st.sidebar.selectbox("Visualization", ["Pareto front", "Action heatmap", "Reward correlations", "Flat results table", "Per-weight tradeoff (radar)"])

# load data early
flat_df = try_load_csv(FILES["flat_csv"])
pareto_df = try_load_csv(FILES["pareto_csv"])

# optional action->drug map
action_drug_df = None
if file_exists(ACTION_DRUG_MAP_CSV):
    try:
        action_drug_df = pd.read_csv(ACTION_DRUG_MAP_CSV)
    except Exception as e:
        st.sidebar.warning(f"Could not load action_to_drug_map.csv: {e}")

# quick summary (top row)
st.header("Polypharmacy MORL — Phase 8 Results")
col1, col2 = st.columns([3,1])
with col1:
    st.markdown("**Artifacts looked for:**")
    for k,v in FILES.items():
        st.write(f"- {k}: `{v.name}` — {'found' if file_exists(v) else 'missing'}")
with col2:
    st.markdown("**Actions**")
    for a,desc in ACTION_MAP.items():
        st.write(f"**{a}** — {desc}")

# --------------------
# Pareto front view
# --------------------
if viz == "Pareto front":
    st.subheader("Pareto front — mean rewards by weight")
    if pareto_df is not None:
        st.dataframe(pareto_df.head(30))
        x_col = st.sidebar.selectbox("X axis", ["efficacy","neg_ddi"], index=0)
        y_col = st.sidebar.selectbox("Y axis", ["neg_ddi","efficacy"], index=1)
        color_col = st.sidebar.selectbox("Color", ["neg_tol","efficacy","neg_ddi"], index=0)
        fig = px.scatter(pareto_df, x=x_col, y=y_col, color=color_col, hover_data=list(pareto_df.columns))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Download CSV of displayed Pareto**")
        csv = pareto_df.to_csv(index=False).encode()
        st.download_button("Download pareto CSV", csv, file_name=f"pareto_{pd.Timestamp.now().strftime('%Y%m%d')}.csv")
    else:
        st.info("Pareto CSV not found. If you have `pareto_front_mean_by_weight.csv`, place it in the Phase8 folder.")

    # show precomputed plot image if present
    if file_exists(FILES["pareto_plot"]):
        st.markdown("**Precomputed Pareto plot**")
        display_image(FILES["pareto_plot"])

# --------------------
# Action heatmap
# --------------------
elif viz == "Action heatmap":
    st.subheader("Action frequency heatmap")
    if file_exists(FILES["heatmap"]):
        display_image(FILES["heatmap"])
        st.write("Heatmap legend: rows = weight / patient buckets, cols = action index")
    else:
        st.info("Heatmap image not found. If you have `action_frequency_heatmap.png` put it in the Phase8 folder.")

    # show top actions from flat results if available
    if flat_df is not None:
        st.markdown("**Top action frequencies (flat results)**")
        if "action" in flat_df.columns:
            freq = flat_df["action"].value_counts().rename_axis("action").reset_index(name="count")
            # map to description
            freq["meaning"] = freq["action"].map(ACTION_MAP).fillna("Unknown")
            st.dataframe(freq)
        else:
            st.info("`action` column not found in flat CSV. Check your flat results file.")

# --------------------
# Reward correlations
# --------------------
elif viz == "Reward correlations":
    st.subheader("Reward correlation matrix")
    if file_exists(FILES["corr"]):
        display_image(FILES["corr"])
    else:
        st.info("Correlation image not found. Attempting to compute from flat results.")
        if flat_df is not None:
            # attempt to compute correlation of reward columns (try to detect reward columns)
            reward_cols = [c for c in flat_df.columns if c.lower() in ("efficacy","neg_ddi","neg_tol","tolerability","ddi","adr")]
            if len(reward_cols) >= 2:
                corr = flat_df[reward_cols].corr()
                fig, ax = plt.subplots(figsize=(5,4))
                sns = __import__("seaborn")
                sns.heatmap(corr, annot=True, ax=ax, cmap="vlag", center=0)
                st.pyplot(fig)
            else:
                st.info("Not enough reward columns found in flat CSV to compute correlation.")
        else:
            st.info("Flat CSV not found; cannot compute correlation.")

# --------------------
# Flat results table
# --------------------
elif viz == "Flat results table":
    st.subheader("Flat results")
    if flat_df is not None:
        st.write("Preview of flat results (first 200 rows)")
        st.dataframe(flat_df.head(200))
        # allow filtering by hadm_id or action
        with st.expander("Filters"):
            if "hadm_id" in flat_df.columns:
                hadm = st.text_input("Filter by hadm_id (exact)", "")
                if hadm:
                    try:
                        filtered = flat_df[flat_df["hadm_id"].astype(str) == hadm]
                        st.dataframe(filtered)
                    except Exception:
                        st.error("Could not filter by hadm_id.")
            if "action" in flat_df.columns:
                actions = sorted(flat_df["action"].unique().tolist())
                sel = st.multiselect("Show actions", actions, default=actions[:3])
                if sel:
                    st.dataframe(flat_df[flat_df["action"].isin(sel)].head(500))
        # download
        csv = flat_df.to_csv(index=False).encode()
        st.download_button("Download flat CSV", csv, file_name="phase8_flat_results.csv")
    else:
        st.info("Flat CSV not found at expected path.")

# --------------------
# Per-weight radar charts (tradeoff)
# --------------------
elif viz == "Per-weight tradeoff (radar)":
    st.subheader("Per-weight tradeoff radar plots")
    if file_exists(FILES["radar"]):
        display_image(FILES["radar"])
    else:
        st.info("Radar image not found, trying to compute from pareto CSV.")
        if pareto_df is not None:
            # assume pareto_df contains columns: weight_*, efficacy, neg_ddi, neg_tol (or similar)
            # allow user to pick a weight row
            st.markdown("Select a weight index to view tradeoff")
            n = len(pareto_df)
            i = st.slider("Weight index (row)", 0, max(0,n-1), 0)
            row = pareto_df.iloc[i]
            # try to build radar
            try:
                categories = ["efficacy","neg_ddi","neg_tol"]
                values = [row[c] for c in categories]
                # radar using plotly
                dfp = pd.DataFrame(dict(r=values, theta=categories))
                fig = px.line_polar(dfp, r="r", theta="theta", line_close=True)
                fig.update_traces(fill='toself')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not make radar from pareto row: {e}")
        else:
            st.info("Pareto CSV not found. Precomputed radar image is also missing.")

# --------------------
# Action-to-drug mapping helper
# --------------------
st.sidebar.markdown("---")
st.sidebar.header("Action → Drug mapping")
st.sidebar.write("If you have a CSV `action_to_drug_map.csv` (action_index,drug_list,notes) place it in the repo root.")
if action_drug_df is not None:
    st.sidebar.dataframe(action_drug_df)
else:
    st.sidebar.write("No action_to_drug_map.csv found. Using default short mappings:")
    for k,v in ACTION_MAP.items():
        st.sidebar.write(f"**{k}** — {v}")

# --------------------
# Footer / exports
# --------------------
st.markdown("---")
st.markdown("**Notes**: This viewer uses precomputed Phase-8 results. To integrate live policy inference in the UI you would need a small inference API (separate) that reconstructs the Q-network(s) or precomputes action frequencies per-weight. Avoid trying to load full training dependencies in the Streamlit app — heavy libs (torch, morl-baselines) cause dependency conflicts on deploy platforms.")
