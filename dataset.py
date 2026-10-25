"""Dataset generation + feature engineering (own metrics + dependency-graph features)."""
import numpy as np
import pandas as pd
import networkx as nx
from config import G, SERVICES, H
from simulator import simulate, DOWN

FEATURES = [
    "err3", "lat3", "cpu", "err_delta5",
    "dep_err_mean", "dep_err_max", "dep_lat_mean", "dep_err_delta5",
    "n_deps", "n_dependents", "n_trans_deps",
]


def generate_runs(n_runs=240, T=100, seed=7):
    rng = np.random.default_rng(seed)
    roots = [s for s in SERVICES if s != "frontend"]
    dfs = []
    for i in range(n_runs):
        root = str(rng.choice(roots))
        fail_at = int(rng.integers(20, 50))
        df = simulate(root, T=T, fail_at=fail_at, seed=int(rng.integers(1_000_000_000)))
        df["run_id"], df["root"], df["fail_at"] = i, root, fail_at
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def _features_one_run(df, graph=G, with_labels=True):
    err = df.pivot(index="t", columns="service", values="error_rate")
    lat = df.pivot(index="t", columns="service", values="latency")
    cpu = df.pivot(index="t", columns="service", values="cpu")
    err3 = err.rolling(3, min_periods=1).mean()
    lat3 = lat.rolling(3, min_periods=1).mean()
    d_err = (err3 - err3.shift(5)).fillna(0)
    status = df.pivot(index="t", columns="service", values="status")

    out = []
    for s in err.columns:
        deps = list(graph.successors(s))
        f = pd.DataFrame({"t": err.index, "service": s})
        f["err3"], f["lat3"], f["cpu"] = err3[s].values, lat3[s].values, cpu[s].values
        f["err_delta5"] = d_err[s].values
        if deps:
            f["dep_err_mean"] = err3[deps].mean(axis=1).values
            f["dep_err_max"] = err3[deps].max(axis=1).values
            f["dep_lat_mean"] = lat3[deps].mean(axis=1).values
            f["dep_err_delta5"] = d_err[deps].mean(axis=1).values
        else:
            for c in ["dep_err_mean", "dep_err_max", "dep_lat_mean", "dep_err_delta5"]:
                f[c] = 0.0
        f["n_deps"] = len(deps)
        f["n_dependents"] = len(nx.ancestors(graph, s))
        f["n_trans_deps"] = len(nx.descendants(graph, s))
        f["status"] = status[s].values
        if with_labels:
            down = (status[s] == DOWN).astype(int)
            fut = down.iloc[::-1].rolling(H, min_periods=1).max().iloc[::-1].shift(-1).fillna(0)
            f["label"] = fut.values.astype(int)
        out.append(f)
    return pd.concat(out, ignore_index=True)


def build_features(df, graph=G, with_labels=True):
    """df may hold many runs (needs run_id) or a single run."""
    if "run_id" not in df.columns:
        return _features_one_run(df, graph, with_labels)
    parts = []
    for rid, g in df.groupby("run_id"):
        f = _features_one_run(g, graph, with_labels)
        f["run_id"] = rid
        parts.append(f)
    return pd.concat(parts, ignore_index=True)
