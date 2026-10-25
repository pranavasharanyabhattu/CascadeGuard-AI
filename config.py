"""Shared configuration: service graph, layout, recovery strategies."""
import networkx as nx

# Edge A -> B means "A depends on B". Failures flow from B up to A.
EDGES = [
    ("frontend", "api"),
    ("api", "auth"), ("api", "payments"), ("api", "search"),
    ("auth", "db"), ("payments", "db"), ("payments", "queue"),
    ("search", "cache"), ("cache", "db"),
]
G = nx.DiGraph(EDGES)
SERVICES = list(G.nodes)

# Fixed layout for the dashboard (frontend on top, db at the bottom)
POS = {
    "frontend": (0, 4), "api": (0, 3),
    "auth": (-2, 2), "payments": (0, 2), "search": (2, 2),
    "queue": (-2, 1), "cache": (2, 1), "db": (0, 0),
}

H = 10               # prediction horizon (minutes)
ERR_THRESH = 0.20    # smoothed error-rate level treated as "anomalous"

# Recovery strategies: cost is in "service-minute" units
STRATEGIES = {
    "none":            dict(cost=0,  label="Do nothing"),
    "restart_root":    dict(cost=10, label="Restart root service"),
    "circuit_breaker": dict(cost=6,  label="Circuit breakers on dependents"),
    "scale_out":       dict(cost=12, label="Scale out dependents"),
    "failover":        dict(cost=20, label="Fail over root to standby"),
}
W_DOWN = 1.0         # loss weight per service-minute DOWN
W_DEGRADED = 0.4     # loss weight per service-minute DEGRADED
