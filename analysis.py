"""Root-cause tracing + recovery-strategy comparison."""
import numpy as np
import pandas as pd
import networkx as nx
from config import G, ERR_THRESH, STRATEGIES, W_DOWN, W_DEGRADED
from simulator import simulate, DOWN, DEGRADED


def _err3(df):
    return df.pivot(index="t", columns="service", values="error_rate").rolling(3, min_periods=1).mean()


def onset_table(df, upto_t=None, graph=G):
    """First minute each service's smoothed error rate crossed the anomaly threshold."""
    e = _err3(df if upto_t is None else df[df.t <= upto_t])
    rows = []
    for s in e.columns:
        hit = e.index[e[s] > ERR_THRESH]
        if len(hit):
            rows.append(dict(t=int(hit[0]), service=s, event="anomaly detected"))
    return pd.DataFrame(rows, columns=["t", "service", "event"]).sort_values("t").reset_index(drop=True)


def detect_root_cause(df, t_now, graph=G):
    """Metrics-only root-cause tracer.

    Candidates = anomalous services none of whose dependencies are anomalous.
    Best candidate = explains the most anomalous services downstream; ties -> earliest onset.
    """
    e = _err3(df[df.t <= t_now])
    anomalous = [s for s in e.columns if e[s].iloc[-1] > ERR_THRESH]
    if not anomalous:
        return None
    onset = {s: int(e.index[e[s] > ERR_THRESH][0]) for s in anomalous}
    cands = [s for s in anomalous if not any(d in anomalous for d in graph.successors(s))]
    if not cands:
        cands = anomalous

    def score(c):
        explained = [a for a in anomalous if a == c or a in nx.ancestors(graph, c)]
        return (len(explained), -onset[c])

    root = max(cands, key=score)
    affected = [a for a in anomalous if a != root and root in nx.descendants(graph, a)]
    edges = set()
    for a in affected:
        path = nx.shortest_path(graph, a, root)
        edges.update(zip(path[:-1], path[1:]))
    return dict(root=root, confidence=round(len(affected) / max(len(anomalous) - 1, 1), 2)
                if len(anomalous) > 1 else 1.0,
                affected=affected, edges=sorted(edges), onset=onset)


def compare_strategies(root, fail_at=30, T=100, n_seeds=30, apply_delay=3):
    """Re-run the same failure under every strategy; average outcomes over seeds."""
    rows = []
    for key, info in STRATEGIES.items():
        down_m, deg_m, peak = [], [], []
        for sd in range(n_seeds):
            d = simulate(root, T=T, fail_at=fail_at, seed=1000 + sd,
                         strategy=key, apply_delay=apply_delay)
            down_m.append((d.status == DOWN).sum())
            deg_m.append((d.status == DEGRADED).sum())
            peak.append(d[d.status == DOWN].groupby("t").size().max() if (d.status == DOWN).any() else 0)
        loss = W_DOWN * np.mean(down_m) + W_DEGRADED * np.mean(deg_m) + info["cost"]
        rows.append(dict(strategy=key, label=info["label"],
                         down_service_min=round(float(np.mean(down_m)), 1),
                         degraded_service_min=round(float(np.mean(deg_m)), 1),
                         peak_services_down=round(float(np.mean(peak)), 1),
                         cost=info["cost"], total_loss=round(float(loss), 1)))
    out = pd.DataFrame(rows).sort_values("total_loss").reset_index(drop=True)
    return out
