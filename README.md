# CascadeGuard
Predict cascading software failures, trace the root cause, and recommend the best recovery strategy.
(ORCA CODER · Track 01 · Software & AI)

## Run
```bash
pip install -r requirements.txt
python train.py          # optional: app trains automatically on first launch
streamlit run app.py
```

## Files
| File | Job |
|---|---|
| config.py | Service dependency graph, layout, strategies, constants |
| simulator.py | Cascade simulator (metrics per service per minute) + recovery strategies |
| dataset.py | Generates many runs; builds own + dependency-graph features and labels |
| train.py | Trains RandomForest, compares with baseline, saves models/ + metrics.json |
| analysis.py | Metrics-only root-cause tracer, incident timeline, strategy comparison |
| app.py | Streamlit live dashboard |
| test_app.py | Headless smoke test of the dashboard |

## Pipeline
graph -> simulator -> features -> model (risk of failing within 10 min) -> root-cause tracer -> recovery simulator -> dashboard

## 3-minute demo script
1. Pick `db`, drag the minute slider across the failure: services turn amber, then red.
2. Show alerts firing *before* services go down (compare the "flagged" vs "DOWN" chart).
3. Point at the gold ring: root cause traced from metrics only, with the red failure path.
4. Show the strategy table: same failure re-simulated, best strategy highlighted.
5. Move "minutes until action" slider: acting earlier (thanks to prediction) cuts loss.
6. Open "Model evidence": AI vs naive baseline on unseen runs.

## Honest limitations (say these before the judges ask)
- Data is simulated, not production logs. Cascade rules are simple probabilities.
- Recovery "optimal" = lowest weighted loss under our cost assumptions (config.py).
