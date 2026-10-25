"""Cascading-failure simulator: produces per-minute metrics per service."""
import numpy as np
import pandas as pd
from config import G

OK, DEGRADED, DOWN = 0, 1, 2


def simulate(root, T=100, fail_at=30, seed=0, strategy="none",
             apply_delay=3, p_spread=0.30, graph=G):
    """Inject a failure at `root` at minute `fail_at` and let it cascade.

    strategy is applied `apply_delay` minutes after the failure starts.
    Returns a DataFrame: t, service, status, latency, error_rate, cpu.
    """
    rng = np.random.default_rng(seed)
    nodes = list(graph.nodes)
    deps = {n: list(graph.successors(n)) for n in nodes}
    status = {n: OK for n in nodes}
    apply_at = fail_at + apply_delay
    rows = []

    for t in range(T):
        prev = dict(status)
        active = strategy != "none" and t >= apply_at
        spread, down_k = p_spread, 0.15
        if active and strategy == "circuit_breaker":
            spread *= 0.3
        if active and strategy == "scale_out":
            spread *= 0.6
            down_k *= 0.4
        restored = strategy == "restart_root" and t >= apply_at + 4

        pressure = {}
        for n in nodes:
            vals = []
            for d in deps[n]:
                s = prev[d]
                if strategy == "failover" and active and d == root:
                    s = OK  # standby is serving dependents
                vals.append(0.0 if s == OK else (0.5 if s == DEGRADED else 1.0))
            bad = sum(vals) / len(vals) if vals else 0.0
            pressure[n] = bad
            cur = prev[n]

            if n == root:
                if t < fail_at or restored:
                    status[n] = OK
                elif strategy == "failover" and active:
                    status[n] = DEGRADED
                else:
                    status[n] = DOWN
                continue

            if cur == OK:
                if rng.random() < spread * bad:
                    status[n] = DEGRADED
            elif cur == DEGRADED:
                if bad > 0:
                    if rng.random() < 0.05 + down_k * bad:
                        status[n] = DOWN
                elif rng.random() < 0.25:
                    status[n] = OK
            else:  # DOWN
                if bad == 0 and rng.random() < 0.20:
                    status[n] = DEGRADED

        for n in nodes:
            # metrics leak a little upstream pressure even while "ok"
            level = status[n] + 0.3 * pressure[n]
            rows.append(dict(
                t=t, service=n, status=status[n],
                latency=rng.normal(100 + 400 * level, 40),
                error_rate=float(np.clip(rng.normal(0.01 + 0.3 * level, 0.05), 0, 1)),
                cpu=float(np.clip(rng.normal(0.4 + 0.2 * level, 0.06), 0, 1)),
            ))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = simulate("db")
    print(df[df.t == 70].set_index("service")[["status", "latency", "error_rate"]])
