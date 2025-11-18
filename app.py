# app.py — Minimal Streamlit demo for gpipd_model_clean.pt inference
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(layout="wide", page_title="Polypharmacy RL Demo")

MODEL_PATH = "gpipd_model_clean.pt"  # make sure this file is in the repo root
ENV_MODULE = "polypharmacy_env"      # file polypharmacy_env.py must contain PolypharmacyEnv class

st.title("Polypharmacy Regimen Optimization — Inference Demo")
st.write("Lightweight Streamlit demo that loads a clean PyTorch checkpoint (no morl-baselines required).")

# -------------------------
# UTIL: Build MLP from state_dict
# -------------------------
def build_mlp_from_state_dict(state_dict):
    """
    state_dict: a dict of parameter tensors for one network.
    Strategy:
      - collect ordered unique prefixes (prefix = key.rsplit('.',1)[0]) in first-seen order
      - for each prefix, expect weight and bias keys: prefix + '.weight' (2D), prefix + '.bias' (1D)
      - create nn.Linear(in_features=weight.shape[1], out_features=weight.shape[0])
      - stack with ReLU between layers (except after last)
      - copy weights/biases into created layers
    Returns:
      nn.Sequential model, final_out_features
    """
    # gather prefixes in order
    prefixes = []
    for k in state_dict.keys():
        if not (k.endswith(".weight") or k.endswith(".bias")):
            continue
        p = k.rsplit(".", 1)[0]
        if p not in prefixes:
            prefixes.append(p)

    if len(prefixes) == 0:
        raise ValueError("No linear-style parameters found in state_dict.")

    layers = []
    # create layers based on weight shapes
    for i, p in enumerate(prefixes):
        w_key = p + ".weight"
        b_key = p + ".bias"
        if w_key not in state_dict:
            raise ValueError(f"Expected weight key {w_key} in state_dict.")
        w = state_dict[w_key]
        if w.dim() != 2:
            raise ValueError(f"Unsupported weight shape for {w_key}: {tuple(w.size())}")
        out_f, in_f = w.size()
        lin = nn.Linear(in_f, out_f)
        # copy params
        lin.weight.data.copy_(w)
        if b_key in state_dict:
            lin.bias.data.copy_(state_dict[b_key])
        else:
            nn.init.zeros_(lin.bias)
        layers.append((f"lin{i}", lin))
        # add nonlinearity except after last
        if i < len(prefixes) - 1:
            layers.append((f"relu{i}", nn.ReLU()))

    seq = nn.Sequential(nn.OrderedDict(layers))
    # determine final out features
    final_out_features = list(state_dict.values())[-1].size(0) if len(state_dict) else None
    return seq, final_out_features

# -------------------------
# Load checkpoint and reconstruct nets
# -------------------------
@st.cache_resource
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}. Put gpipd_model_clean.pt in repo root.")
    ckpt = torch.load(path, map_location="cpu")
    if "q_nets" not in ckpt:
        raise KeyError("Checkpoint missing 'q_nets' key. Expect a dict with 'q_nets' and 'weight_support'.")
    q_states = ckpt["q_nets"]
    weight_support = ckpt.get("weight_support", None)
    if weight_support is None:
        raise KeyError("Checkpoint missing 'weight_support' key.")

    # ensure weight_support is numpy array
    weight_support = np.array(weight_support)

    # reconstruct models
    models = []
    out_sizes = []
    for idx, sd in enumerate(q_states):
        model, final_out = build_mlp_from_state_dict(sd)
        models.append(model)
        out_sizes.append(final_out)

    # infer reward dim and action dim:
    # we know weight_support is shape (k, reward_dim)
    reward_dim = int(weight_support.shape[1])
    # final_out should equal action_dim * reward_dim (common design)
    # derive action_dim from first model
    action_dim = None
    if len(out_sizes) > 0 and out_sizes[0] is not None:
        if out_sizes[0] % reward_dim == 0:
            action_dim = out_sizes[0] // reward_dim
        else:
            # fallback: assume action_dim=out_sizes[0] and reward_dim=1
            action_dim = out_sizes[0]
            st.warning("Could not cleanly deduce action_dim from network output — assuming action_dim = final_out_features")

    return models, weight_support, int(reward_dim), int(action_dim)

# -------------------------
# Safe wrapper inference
# -------------------------
def evaluate_action(models, obs, w):
    """
    models: list of reconstructed q-nets (we'll use first)
    obs: 1D numpy array (observation)
    w: 1D numpy array (weight vector, length reward_dim)
    Strategy:
      - concat obs and w to a single input
      - run forward on the first model (or ensemble average if multiple)
      - get output vector of length action_dim * reward_dim
      - reshape to (action_dim, reward_dim)
      - scalarize each action by dot(reward_vector, w)
      - return chosen action index and per-action scalarized values and per-action reward vectors
    """
    device = torch.device("cpu")
    inp = np.concatenate([np.asarray(obs).ravel(), np.asarray(w).ravel()]).astype(np.float32)
    x = torch.from_numpy(inp).to(device)
    # if multiple models available, average their raw outputs
    outs = []
    for m in models:
        m.eval()
        with torch.no_grad():
            out = m(x)  # 1D tensor
            outs.append(out.cpu().numpy())
    mean_out = np.mean(np.stack(outs, axis=0), axis=0)
    # sizes
    total = mean_out.size
    reward_dim = int(w.size)
    if reward_dim == 0:
        raise ValueError("weight vector empty.")
    if total % reward_dim != 0:
        # fallback: assume reward_dim==1
        action_dim = total
        reward_dim = 1
    else:
        action_dim = total // reward_dim
    reshaped = mean_out.reshape(action_dim, reward_dim)
    # scalarize
    scalar_vals = reshaped.dot(w)
    best_action = int(np.argmax(scalar_vals))
    return best_action, scalar_vals, reshaped

# -------------------------
# Load model
# -------------------------
try:
    models, weight_support, reward_dim, action_dim = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

st.success(f"Loaded model — detected reward_dim={reward_dim}, (approx) action_dim={action_dim}")
st.write("Weight support size:", weight_support.shape)

# -------------------------
# Load environment
# -------------------------
try:
    # user must provide polypharmacy_env.py with PolypharmacyEnv class
    from polypharmacy_env import PolypharmacyEnv
    env = PolypharmacyEnv()
    st.write("Environment loaded:", env)
except Exception as e:
    st.warning(f"Could not import PolypharmacyEnv automatically (still OK). Error: {e}")
    env = None

# -------------------------
# Simple action -> human label mapping (user edit as required)
# -------------------------
default_action_map = {i: f"action_{i}" for i in range(max(1, action_dim))}
st.sidebar.header("Action labels (edit)")
for i in range(min(10, action_dim)):
    # let user override labels
    lbl = st.sidebar.text_input(f"Action {i} label", value=default_action_map.get(i, f"action_{i}"))
    default_action_map[i] = lbl

# -------------------------
# Weights sliders
# -------------------------
st.sidebar.header("Policy weight sliders (sum will be normalized)")
num_weights = reward_dim
weights = []
for i in range(num_weights):
    wv = st.sidebar.slider(f"w[{i}]", 0.0, 1.0, float(1.0 / num_weights))
    weights.append(wv)
weights = np.array(weights, dtype=np.float32)
if weights.sum() == 0:
    weights = np.ones_like(weights) / weights.size
else:
    weights = weights / weights.sum()
st.sidebar.write("Normalized weights:", weights.tolist())

# -------------------------
# Single-step sim UI
# -------------------------
st.header("Single-step inference")
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("Environment / Observation")
    if env is not None:
        obs, _ = env.reset()
        st.write("Sample observation (from env.reset()):", obs.tolist())
    else:
        # let user upload an observation or enter manually
        obs_text = st.text_area("Enter observation as comma-separated values (or upload CSV below):", value="")
        if obs_text.strip():
            obs = np.fromstring(obs_text, sep=",", dtype=np.float32)
        else:
            obs = np.zeros(max(1, models[0][0].in_features - reward_dim), dtype=np.float32)
            st.write("Using default zero observation:", obs.tolist())

    if st.button("Run inference (single step)"):
        try:
            action, scalar_vals, reward_vectors = evaluate_action(models, obs, weights)
            st.write("Selected action index:", action)
            st.write("Action label:", default_action_map.get(action, str(action)))
            st.write("Scalarized action values (per-action):", scalar_vals.tolist())
            st.write("Per-action reward vectors (each row):")
            st.write(pd.DataFrame(reward_vectors))
        except Exception as e:
            st.error(f"Inference failed: {e}")

with col2:
    st.subheader("Batch evaluation (upload CSV of observations)")
    uploaded = st.file_uploader("Upload CSV with rows = observations (optional)", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write("Preview:", df.head())
        # assume each row is an observation vector
        obs_rows = df.values.astype(np.float32)
        actions = []
        for r in obs_rows:
            a, s, rv = evaluate_action(models, r, weights)
            actions.append({"action": int(a), "scalar": float(s[a])})
        out_df = pd.DataFrame(actions)
        st.write("Predicted actions for uploaded rows:")
        st.dataframe(out_df)
        st.download_button("Download predictions CSV", out_df.to_csv(index=False), file_name="predictions.csv")

# -------------------------
# Optional Pareto / plots (visualization)
# -------------------------
st.header("Visualizations / Pareto (optional)")
if st.button("Show sample Pareto scatter from weight_support + model"):
    # Evaluate model on weight_support set (mean returns)
    results = []
    for w in weight_support:
        # run a few episodes or use a single obs sample
        if env is not None:
            # perform one eval rollout and average
            obs, _ = env.reset()
        else:
            obs = np.zeros(models[0][0].in_features - reward_dim, dtype=np.float32)
        a, scalar_vals, reward_vectors = evaluate_action(models, obs, np.array(w))
        # store the greedy reward vector for chosen action
        chosen_reward = reward_vectors[a]
        results.append({"w": list(w), "action": a, "scalar": float(scalar_vals[a]), **{f"r{i}": float(chosen_reward[i]) for i in range(len(chosen_reward))}})
    df = pd.DataFrame(results)
    st.write(df.head())
    fig, ax = plt.subplots(figsize=(6, 4))
    if "r0" in df.columns and "r1" in df.columns:
        ax.scatter(df["r0"], df["r1"], c=df["scalar"])
        ax.set_xlabel("r0")
        ax.set_ylabel("r1")
        st.pyplot(fig)
    else:
        st.write("Not enough reward dims to scatter.")

st.write("Done. Edit action labels in the sidebar to map numeric actions to clinical descriptions.")
