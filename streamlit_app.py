from pathlib import Path
import sys
import traceback
import numpy as np
import pandas as pd
import streamlit as st

# Run this file from the existing rl/ project root, where `environment/` and
# `ppo_fault_diagnosis.zip` already exist.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="PPO Fault Diagnosis Lab", page_icon="🛠️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1450px;}
[data-testid="stMetric"] {background: rgba(128,128,128,.07); padding: 14px 16px; border-radius: 12px; border: 1px solid rgba(128,128,128,.18);}
div.stButton > button {border-radius: 9px; min-height: 2.7rem;}
.small-note {opacity: .78; font-size: .9rem;}
</style>
""", unsafe_allow_html=True)

st.title("🛠️ Cost-Aware Active Fault Diagnosis & Recovery")
st.caption("Interactive demonstration · PPO agent · Partially observable distributed IT system")

FAULTS = ["CPU overload", "Memory leak", "Network degradation", "Database failure", "Service failure"]
METRICS = ["CPU", "Memory", "Request rate", "Response time", "Error rate", "Queue length", "Availability", "Network latency", "Packet loss", "DB latency", "DB failure rate"]
ACTIONS = {
    0: "Request Telemetry", 1: "Request Health Probe", 2: "Inspect Logs", 3: "Run Deep Diagnostic",
    4: "Restart Service", 5: "Reroute Traffic", 6: "Isolate Component", 7: "Reduce Load", 8: "Rollback", 9: "Stop System",
    10: "Verify Recovery / Wait"
}

@st.cache_resource(show_spinner="Loading trained PPO model...")
def load_model(model_path: str):
    from stable_baselines3 import PPO
    return PPO.load(model_path, device="cpu")

def make_env(seed: int):
    from environment.rl_environment import FaultDiagnosisEnv
    return FaultDiagnosisEnv(seed=seed)

def fault_label(raw):
    if raw is None:
        return "Unknown"
    s = getattr(raw, "name", str(raw)).replace("_", " ").title()
    aliases = {"Cpu Overload": "CPU overload", "Db Failure": "Database failure", "Database Failure": "Database failure", "Network Degradation": "Network degradation", "Memory Leak": "Memory leak", "Service Failure": "Service failure"}
    return aliases.get(s, s)

def current_state(env, obs):
    state = env.system.get_state()
    raw = [state.cpu, state.memory, state.request_rate, state.response_time, state.error_rate,
           state.queue_length, state.availability, state.network_latency, state.packet_loss,
           state.db_latency, state.db_failure_rate]
    return {"metrics": {k: float(v) for k, v in zip(METRICS, raw)},
            "beliefs": {k: float(v) for k, v in zip(FAULTS, np.asarray(obs)[11:16])},
            "uncertainty": float(obs[16]),
            "budget": float(getattr(env, "remaining_budget", 0)),
            "cost": float(getattr(env, "accumulated_cost", 0)),
            "queries": int(getattr(env, "query_count", 0)),
            "elapsed": float(getattr(env, "elapsed_time", 0)),
            "pending_verification": bool(getattr(env, "pending_verification", False)),
            "verification_streak": int(getattr(env, "healthy_verification_streak", 0))}

def start_episode(seed: int):
    env = make_env(seed)
    obs, info = env.reset(seed=seed)
    model_path = ROOT / "ppo_fault_diagnosis_v2.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"Could not find {model_path}. Train the v2 model first with train_ppo.py; expected file: ppo_fault_diagnosis_v2.zip.")
    model = load_model(str(model_path))
    st.session_state.env = env
    st.session_state.model = model
    st.session_state.obs = obs
    st.session_state.done = False
    st.session_state.episode_seed = seed
    st.session_state.hidden_fault = info.get("hidden_fault", getattr(getattr(env, "fault_system", None), "fault", None))
    st.session_state.history = []
    st.session_state.total_reward = 0.0
    st.session_state.last_info = info
    st.session_state.last_reward = None
    st.session_state.action_count = 0
    st.session_state.last_action = None
    st.session_state.last_state = current_state(env, obs)

def take_step():
    if st.session_state.get("done", True):
        return
    env, model, obs = st.session_state.env, st.session_state.model, st.session_state.obs
    action, _ = model.predict(obs, deterministic=True)
    action = int(np.asarray(action).item())
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = bool(terminated or truncated)
    state = current_state(env, next_obs)
    dresult = info.get("diagnostic_result")
    rresult = info.get("recovery_result")
    event = {
        "Step": len(st.session_state.history) + 1,
        "Action": ACTIONS.get(action, f"Action {action}"),
        "Type": "Diagnostic" if action < 4 else ("Recovery" if action < 10 else "Verification"),
        "Reward": float(reward),
        "Diagnostic cost": float((dresult or {}).get("cost", 0) or 0),
        "Delay": float((dresult or {}).get("delay", (rresult or {}).get("delay", 0)) or 0),
        "Outcome": str((dresult or {}).get("result", (rresult or {}).get("result", "—"))),
        "Budget remaining": state["budget"],
        "Uncertainty": state["uncertainty"],
    }
    if rresult:
        event["Outcome"] = str(rresult.get("message", rresult.get("success", "Recovery attempted")))
        if "success" in rresult:
            event["Outcome"] = "Intervention applied; verification required" if rresult["success"] else "Failed / unsuccessful"
    if info.get("verification_performed"):
        event["Outcome"] = "Healthy check passed" if info.get("system_healthy_this_check") else "Health check failed"
    if info.get("verified_recovery"):
        event["Outcome"] = "Recovery verified (2 healthy checks)"
    st.session_state.history.append(event)
    st.session_state.obs = next_obs
    st.session_state.done = done
    st.session_state.total_reward += float(reward)
    st.session_state.last_info = info
    st.session_state.last_reward = float(reward)
    st.session_state.action_count += 1
    st.session_state.last_action = ACTIONS.get(action, str(action))
    st.session_state.last_state = state

# Sidebar controls
with st.sidebar:
    st.header("Simulation controls")
    seed = st.number_input("Episode seed", min_value=0, max_value=999999, value=42, step=1)
    st.caption("The agent receives a 23-dimensional observation, not the hidden fault label.")
    if st.button("🔄 Start / reset episode", use_container_width=True, type="primary"):
        try:
            start_episode(int(seed))
            st.rerun()
        except Exception as e:
            st.error(f"Could not initialize the project environment: {e}")
            st.code(traceback.format_exc())
    if st.button("▶ Run one PPO decision", use_container_width=True, disabled=st.session_state.get("done", True) or "env" not in st.session_state):
        try:
            take_step()
            st.rerun()
        except Exception as e:
            st.error(f"PPO step failed: {e}")
            st.code(traceback.format_exc())
    if st.button("⏩ Run episode to completion", use_container_width=True, disabled=st.session_state.get("done", True) or "env" not in st.session_state):
        try:
            safety_limit = 35
            while not st.session_state.done and len(st.session_state.history) < safety_limit:
                take_step()
            st.rerun()
        except Exception as e:
            st.error(f"Episode run failed: {e}")
            st.code(traceback.format_exc())
    st.divider()
    st.markdown("**Action mapping used by the environment**")
    st.caption("0–3: diagnostics · 4–9: recovery actions · 10: verify recovery / wait")

if "env" not in st.session_state:
    st.info("Click **Start / reset episode** in the sidebar to begin a live episode using your saved PPO model.")
    st.markdown("### What this demo shows")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**1 · Observe**\n\nSystem metrics and fault-belief probabilities")
    c2.markdown("**2 · Decide**\n\nThe trained PPO policy selects one of 11 actions")
    c3.markdown("**3 · Verify**\n\nRecovery must pass two consecutive health checks")
    st.stop()

state = st.session_state.last_state
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Episode status", "Completed" if st.session_state.done else "Running")
c2.metric("Steps", st.session_state.action_count)
c3.metric("Cumulative reward", f"{st.session_state.total_reward:.2f}")
c4.metric("Diagnostic budget", f"{state['budget']:.1f} / 10")
c5.metric("Accumulated cost", f"{state['cost']:.1f}")

left, right = st.columns([1.15, 1])
with left:
    st.subheader("Live system metrics")
    metric_df = pd.DataFrame({"Metric": list(state["metrics"].keys()), "Value": list(state["metrics"].values())})
    st.dataframe(metric_df, hide_index=True, use_container_width=True, height=365)
with right:
    st.subheader("Agent's fault beliefs")
    belief_df = pd.DataFrame({"Possible fault": list(state["beliefs"].keys()), "Belief": list(state["beliefs"].values())})
    st.bar_chart(belief_df.set_index("Possible fault"), y="Belief", use_container_width=True)
    st.progress(min(1.0, max(0.0, state["uncertainty"])), text=f"Normalized uncertainty: {state['uncertainty']:.3f}")
    st.caption(f"Recovery verification pending: {state['pending_verification']} · Consecutive healthy checks: {state['verification_streak']}/2")

st.subheader("Latest decision")
if st.session_state.last_action is None:
    st.write("No decision yet. Select **Run one PPO decision** to observe the agent step by step.")
else:
    a, b, c = st.columns(3)
    a.metric("PPO action", st.session_state.last_action)
    b.metric("Last-step reward", f"{st.session_state.last_reward:.2f}")
    c.metric("Elapsed time", f"{state['elapsed']:.1f}")
    info = st.session_state.last_info or {}
    if info.get("diagnostic_result"):
        st.write("**Diagnostic output**")
        st.json(info["diagnostic_result"])
    if info.get("recovery_result"):
        st.write("**Recovery output**")
        st.json(info["recovery_result"])

st.subheader("Episode action log")
if st.session_state.history:
    st.dataframe(pd.DataFrame(st.session_state.history), hide_index=True, use_container_width=True)
else:
    st.caption("Actions will appear here as the agent runs.")

if st.session_state.done:
    if st.session_state.last_info.get("verified_recovery"):
        st.success("Recovery verified after two consecutive healthy checks.")
    elif st.session_state.last_info.get("safe_termination"):
        st.warning("Episode ended by safe shutdown; service was not restored.")
    elif st.session_state.last_info.get("failure"):
        st.error("Episode ended because the system reached a critical failure condition.")
    else:
        st.info("Episode ended or reached its time limit. Inspect the final outcome below.")
    reveal = st.toggle("Reveal hidden fault (after episode)", value=False)
    if reveal:
        st.info(f"Environment's hidden fault: **{fault_label(st.session_state.hidden_fault)}**")
    st.caption("This is the simulated environment's ground truth; it was not supplied as part of the PPO observation.")
