# PPO Fault Diagnosis v2 — Recovery Verification

This is an updated version of the existing environment, not a real production remediation system. It remains a simulator for a distributed IT service.

## What changed
- The PPO action space now has 11 actions: four diagnostic sources, six recovery actions, and one explicit `VERIFY_RECOVERY / WAIT` action.
- Observation size changes from 21 to 23. Two new features indicate whether recovery verification is pending and the number of consecutive healthy checks.
- A successful recovery intervention no longer ends the episode immediately. The environment requires two consecutive healthy checks before awarding verified recovery.
- The simulator continues evolving during diagnostic-source delay.
- Diagnostic budget exhaustion no longer prevents recovery after the budget is spent; it only prevents unaffordable queries.
- Evaluation reports verified recovery separately from safe shutdown and critical failure.

## Important
The old `ppo_fault_diagnosis.zip` was trained for the old 21-observation / 10-action environment. **Do not load it with this version.** Retrain a fresh compatible model.

## Installation
Use the same Python environment you intend to run Streamlit with:
```powershell
python -m pip install numpy gymnasium stable-baselines3 streamlit pandas
```

## Recommended folder layout
Copy these files into your existing `rl` project folder, replacing its `environment` folder:
```text
rl/
  environment/
    __init__.py (optional)
    system_model.py
    faults.py
    diagnostics.py
    recovery.py
    rl_environment.py
  train_ppo.py
  evaluate_ppo.py
  streamlit_app.py
```
Keep your old model as a backup. The new training script saves `ppo_fault_diagnosis_v2.zip`.

## Train and evaluate
```powershell
python train_ppo.py
python evaluate_ppo.py
python -m streamlit run streamlit_app.py
```
Training uses 200,000 timesteps and may take time on a CPU. The environment is stochastic, so outcomes and episode lengths vary. The policy is not guaranteed to use every diagnostic source; inspect per-fault evaluation and compare against baselines before claiming superiority.

## Action map
| Action | Meaning |
|---:|---|
| 0 | Request telemetry |
| 1 | Request health probe |
| 2 | Inspect logs |
| 3 | Deep diagnostic |
| 4 | Restart service |
| 5 | Reroute traffic |
| 6 | Isolate component |
| 7 | Reduce load |
| 8 | Rollback |
| 9 | Safe stop |
| 10 | Verify recovery / wait |

## Research caveat
The system-health thresholds and fault/recovery probabilities are simulation assumptions, not empirically calibrated operational SLOs. In a research report, document them and perform sensitivity analysis. The simulator is not connected to or authorized to operate a real service.
