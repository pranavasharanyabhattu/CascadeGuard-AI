"""Train the failure-prediction model, compare against a naive baseline, save metrics."""
import json, os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from config import H, ERR_THRESH
from dataset import generate_runs, build_features, FEATURES
from simulator import DOWN

MODEL_PATH = "models/model.joblib"
METRICS_PATH = "models/metrics.json"


def _prf(y, pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return dict(precision=round(float(p), 3), recall=round(float(r), 3), f1=round(float(f), 3))


def _event_metrics(frame, alert_col):
    """Per (run, service) that fails: was it flagged before failure, and how early?"""
    leads, detected, n_fail, quiet, quiet_alarm = [], 0, 0, 0, 0
    for _, g in frame.groupby(["run_id", "service"]):
        g = g.sort_values("t")
        down_t = g.loc[g.status == DOWN, "t"]
        alerts = g[(g[alert_col]) & (g.status != DOWN)]
        if len(down_t):
            n_fail += 1
            fd = int(down_t.iloc[0])
            early = alerts[(alerts.t < fd) & (alerts.t >= fd - H)]
            if len(early):
                detected += 1
                leads.append(fd - int(early.t.min()))
        else:
            quiet += 1
            quiet_alarm += int(len(alerts) > 0)
    return dict(
        detection_rate=round(detected / max(n_fail, 1), 3),
        mean_lead_minutes=round(float(np.mean(leads)) if leads else 0.0, 2),
        false_alarm_rate=round(quiet_alarm / max(quiet, 1), 3),
    )


def main(n_runs=240, seed=7):
    os.makedirs("models", exist_ok=True)
    df = generate_runs(n_runs=n_runs, seed=seed)
    feats = build_features(df)
    feats["eligible"] = feats.status != DOWN   # only predict for not-yet-down services

    rng = np.random.default_rng(seed)
    ids = rng.permutation(feats.run_id.unique())
    n = len(ids)
    tr, va, te = ids[: int(.6 * n)], ids[int(.6 * n): int(.8 * n)], ids[int(.8 * n):]
    train, val, test = (feats[feats.run_id.isin(s)].copy() for s in (tr, va, te))
    train, val = train[train.eligible], val[val.eligible]

    model = RandomForestClassifier(
        n_estimators=150, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", n_jobs=-1, random_state=seed)
    model.fit(train[FEATURES], train.label)

    # choose alert threshold on validation set (best F1)
    pv = model.predict_proba(val[FEATURES])[:, 1]
    best_thr, best_f1 = 0.5, -1
    for thr in np.arange(0.2, 0.91, 0.05):
        f1 = _prf(val.label, pv >= thr)["f1"]
        if f1 > best_f1:
            best_thr, best_f1 = float(round(thr, 2)), f1

    test["proba"] = model.predict_proba(test[FEATURES])[:, 1]
    test["model_alert"] = test.proba >= best_thr
    test["base_alert"] = test.err3 > ERR_THRESH   # naive: "own error rate is high"
    test_elig = test[test.eligible]

    metrics = dict(
        n_runs=n_runs, horizon_minutes=H, threshold=best_thr,
        model=dict(**_prf(test_elig.label, test_elig.model_alert), **_event_metrics(test, "model_alert")),
        baseline=dict(**_prf(test_elig.label, test_elig.base_alert), **_event_metrics(test, "base_alert")),
        feature_importance={f: round(float(i), 4) for f, i in
                            sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])},
    )
    joblib.dump(dict(model=model, threshold=best_thr), MODEL_PATH)
    with open(METRICS_PATH, "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(pd.DataFrame({"model": metrics["model"], "baseline": metrics["baseline"]}))
    print("threshold:", best_thr)
    return metrics


if __name__ == "__main__":
    main()
