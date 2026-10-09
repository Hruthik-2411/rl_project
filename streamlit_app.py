from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="PPO Fault Diagnosis Lab",
    page_icon="🛠️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.3rem; padding-bottom: 2rem; max-width: 1500px;}
    [data-testid="stMetric"] {
        background: rgba(128,128,128,.07);
        padding: 12px 15px;
        border-radius: 10px;
        border: 1px solid rgba(128,128,128,.18);
    }
    div.stButton > button {border-radius: 9px; min-height: 2.6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛠️ Cost-Aware Active Fault Diagnosis & Recovery")
st.caption("Live simulation · PPO policy · Partially observable distributed IT system")

FAULTS = [
    "CPU overload",
    "Memory leak",
    "Network degradation",
    "Database failure",
    "Service failure",
]
METRICS = [
    "CPU", "Memory", "Request rate", "Response time", "Error rate",
    "Queue length", "Availability", "Network latency", "Packet loss",
    "DB latency", "DB failure rate",
]
ACTIONS = {
    0: "Request Telemetry",
    1: "Request Health Probe",
    2: "Inspect Logs",
    3: "Run Deep Diagnostic",
    4: "Restart Service",
    5: "Reroute Traffic",
    6: "Isolate Component",
    7: "Reduce Load",
    8: "Rollback",
    9: "Stop System",
    10: "Verify Recovery / Wait",
}
ACTION_COSTS = {
    0: 1.0,
    1: 2.0,
    2: 3.0,
    3: 6.0,
}
MODEL_PATH = ROOT / "ppo_fault_diagnosis_v4.zip"


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
    name = getattr(raw, "name", str(raw)).replace("_", " ").title()
    aliases = {
        "Cpu Overload": "CPU overload",
        "Db Failure": "Database failure",
        "Database Failure": "Database failure",
        "Network Degradation": "Network degradation",
        "Memory Leak": "Memory leak",
        "Service Failure": "Service failure",
    }
    return aliases.get(name, name)


def get_state(env, obs):
    s = env.system.get_state()
    metric_values = [
        s.cpu, s.memory, s.request_rate, s.response_time, s.error_rate,
        s.queue_length, s.availability, s.network_latency, s.packet_loss,
        s.db_latency, s.db_failure_rate,
    ]
    obs = np.asarray(obs, dtype=np.float32)
    return {
        "metrics": {name: float(value) for name, value in zip(METRICS, metric_values)},
        "beliefs": {name: float(value) for name, value in zip(FAULTS, obs[11:16])},
        "uncertainty": float(obs[16]),
        "budget": float(env.remaining_budget),
        "initial_budget": float(env.initial_budget),
        "cost": float(env.accumulated_cost),
        "queries": int(env.query_count),
        "elapsed": float(env.elapsed_time),
        "pending_verification": bool(env.pending_verification),
        "verification_streak": int(env.healthy_verification_streak),
    }


def reset_episode(seed: int):
    env = make_env(seed)
    obs, info = env.reset(seed=seed)
    st.session_state.env = env
    st.session_state.obs = obs
    st.session_state.done = False
    st.session_state.episode_seed = seed
    st.session_state.hidden_fault = info.get("hidden_fault")
    st.session_state.history = []
    st.session_state.total_reward = 0.0
    st.session_state.last_info = info
    st.session_state.last_reward = None
    st.session_state.action_count = 0
    st.session_state.last_action = None
    st.session_state.last_state = get_state(env, obs)


def execute_action(action: int, source_label: str):
    """Run one genuine action through the environment and log its returned values."""
    if st.session_state.get("done", True):
        st.warning("The episode has ended. Reset it to continue.")
        return

    env = st.session_state.env
    obs = st.session_state.obs
    next_obs, reward, terminated, truncated, info = env.step(int(action))
    done = bool(terminated or truncated)
    state = get_state(env, next_obs)

    diagnostic = info.get("diagnostic_result") or {}
    recovery = info.get("recovery_result") or {}

    if action < 4:
        outcome = diagnostic.get("reason") or diagnostic.get("result") or (
            "Diagnostic action returned no result"
        )
        action_type = "Diagnostic"
        diagnostic_cost = float(diagnostic.get("cost", 0.0) or 0.0)
        delay = int(diagnostic.get("delay", 0) or 0)
    elif action < 10:
        outcome = recovery.get("message", recovery.get("result", "Recovery attempted"))
        if "success" in recovery:
            outcome = (
                "Intervention applied; verification required"
                if recovery["success"]
                else "Failed / unsuccessful"
            )
        action_type = "Recovery"
        diagnostic_cost = 0.0
        delay = int(recovery.get("delay", 0) or 0)
    else:
        action_type = "Verification"
        diagnostic_cost = 0.0
        delay = 0
        if info.get("verification_performed"):
            outcome = (
                "Healthy check passed"
                if info.get("system_healthy_this_check")
                else "Health check failed"
            )
        else:
            outcome = "No pending recovery to verify"

    if info.get("verified_recovery"):
        outcome = "Recovery verified (2 healthy checks)"
    elif info.get("safe_termination"):
        outcome = "System safely stopped"
    elif info.get("failure"):
        outcome = "Critical system failure"

    event = {
        "Step": len(st.session_state.history) + 1,
        "Decision source": source_label,
        "Action": ACTIONS.get(action, f"Action {action}"),
        "Type": action_type,
        "Reward": float(reward),
        "Diagnostic cost": diagnostic_cost,
        "Delay": delay,
        "Outcome": str(outcome),
        "Budget remaining": state["budget"],
        "Queries": state["queries"],
        "Uncertainty": state["uncertainty"],
    }
    st.session_state.history.append(event)
    st.session_state.obs = next_obs
    st.session_state.done = done
    st.session_state.total_reward += float(reward)
    st.session_state.last_info = info
    st.session_state.last_reward = float(reward)
    st.session_state.action_count += 1
    st.session_state.last_action = ACTIONS.get(action, str(action))
    st.session_state.last_state = state
    st.session_state.last_source = source_label


if "env" not in st.session_state:
    st.session_state.done = True
    st.session_state.history = []
    st.session_state.action_count = 0
    st.session_state.total_reward = 0.0
    st.session_state.last_action = None

with st.sidebar:
    st.header("Simulation controls")
    seed = st.number_input("Episode seed", min_value=0, max_value=999999, value=42, step=1)
    mode = st.radio(
        "Decision mode",
        ["Trained PPO agent", "Manual environment test"],
        help=(
            "PPO mode uses model.predict(). Manual mode is a transparent debugging/demo "
            "control and does not represent the learned policy."
        ),
    )
    st.caption("Observation: 23 values · Action space: 11 discrete actions")

    if st.button("🔄 Start / reset episode", use_container_width=True, type="primary"):
        try:
            reset_episode(int(seed))
            st.rerun()
        except Exception as exc:
            st.error(f"Could not initialize the environment: {exc}")
            st.code(traceback.format_exc())

    if mode == "Trained PPO agent":
        if not MODEL_PATH.exists():
            st.error(f"Model not found: {MODEL_PATH.name}")
        else:
            st.caption(f"Model: {MODEL_PATH.name}")
        if st.button(
            "▶ Run one PPO decision",
            use_container_width=True,
            disabled=st.session_state.get("done", True) or "env" not in st.session_state or not MODEL_PATH.exists(),
        ):
            try:
                model = load_model(str(MODEL_PATH))
                action, _ = model.predict(st.session_state.obs, deterministic=True)
                execute_action(int(np.asarray(action).item()), "PPO")
                st.rerun()
            except Exception as exc:
                st.error(f"PPO step failed: {exc}")
                st.code(traceback.format_exc())

        if st.button(
            "⏩ Run PPO episode to completion",
            use_container_width=True,
            disabled=st.session_state.get("done", True) or "env" not in st.session_state or not MODEL_PATH.exists(),
        ):
            try:
                model = load_model(str(MODEL_PATH))
                safety_limit = 35
                while not st.session_state.done and len(st.session_state.history) < safety_limit:
                    action, _ = model.predict(st.session_state.obs, deterministic=True)
                    execute_action(int(np.asarray(action).item()), "PPO")
                st.rerun()
            except Exception as exc:
                st.error(f"Episode run failed: {exc}")
                st.code(traceback.format_exc())
    else:
        manual_action_name = st.selectbox(
            "Choose an environment action",
            options=list(ACTIONS.values()),
            index=0,
        )
        manual_action = next(k for k, v in ACTIONS.items() if v == manual_action_name)
        if manual_action < 4:
            st.caption(f"Configured diagnostic cost: {ACTION_COSTS[manual_action]:g}")
        if st.button(
            "Execute selected action",
            use_container_width=True,
            disabled=st.session_state.get("done", True) or "env" not in st.session_state,
        ):
            try:
                execute_action(manual_action, "Manual test (not PPO)")
                st.rerun()
            except Exception as exc:
                st.error(f"Environment action failed: {exc}")
                st.code(traceback.format_exc())

    st.divider()
    st.markdown("**Action mapping**")
    st.caption("0–3: diagnostics · 4–9: recovery/stop · 10: verify/wait")
    st.caption("Only diagnostic actions consume diagnostic budget.")


if "env" not in st.session_state:
    st.info("Click **Start / reset episode** in the sidebar to initialize a live simulation.")
    st.markdown("### Demo flow")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**1 · Observe**\n\nView live system telemetry and fault beliefs.")
    c2.markdown("**2 · Decide**\n\nRun the PPO policy or explicitly test an environment action.")
    c3.markdown("**3 · Verify**\n\nRecovery is confirmed only after two consecutive healthy checks.")
    st.stop()

env = st.session_state.env
state = st.session_state.last_state

st.subheader("Episode overview")
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Episode status", "Completed" if st.session_state.done else "Running")
m2.metric("Steps", st.session_state.action_count)
m3.metric("Cumulative reward", f"{st.session_state.total_reward:.2f}")
m4.metric("Diagnostic budget", f"{state['budget']:.1f} / {state['initial_budget']:.0f}")
m5.metric("Diagnostic cost", f"{state['cost']:.1f}")
m6.metric("Diagnostic queries", state["queries"])

# Explicitly show the current accounting; frontend never deducts budget itself.
st.progress(
    max(0.0, min(1.0, state["budget"] / max(state["initial_budget"], 1.0))),
    text=f"Budget remaining: {state['budget']:.1f} of {state['initial_budget']:.1f}",
)
if st.session_state.get("last_source") == "PPO" and state["queries"] == 0 and st.session_state.action_count > 0:
    st.info(
        "The PPO policy has not selected a diagnostic action in this episode yet. "
        "Therefore, diagnostic budget remains unchanged. Use Manual environment test "
        "only to test budget accounting; those actions are not PPO decisions."
    )

left, right = st.columns([1.1, 1])
with left:
    st.subheader("Live system metrics")
    metric_df = pd.DataFrame(
        {"Metric": list(state["metrics"].keys()), "Value": list(state["metrics"].values())}
    )
    st.dataframe(metric_df, hide_index=True, use_container_width=True, height=365)
with right:
    st.subheader("Agent fault beliefs")
    belief_df = pd.DataFrame(
        {"Possible fault": list(state["beliefs"].keys()), "Belief": list(state["beliefs"].values())}
    )
    st.bar_chart(belief_df.set_index("Possible fault"), y="Belief", use_container_width=True)
    st.progress(
        max(0.0, min(1.0, state["uncertainty"])),
        text=f"Normalized uncertainty: {state['uncertainty']:.3f}",
    )
    st.caption(
        f"Recovery verification pending: {state['pending_verification']} · "
        f"Consecutive healthy checks: {state['verification_streak']}/2"
    )

st.subheader("Latest decision")
if st.session_state.last_action is None:
    st.write("No action taken yet. Use the sidebar to run a PPO decision or a manual test.")
else:
    a, b, c = st.columns(3)
    a.metric("Action", st.session_state.last_action)
    b.metric("Last-step reward", f"{st.session_state.last_reward:.2f}")
    c.metric("Elapsed time", f"{state['elapsed']:.1f}")
    st.caption(f"Decision source: {st.session_state.get('last_source', 'Unknown')}")
    last_info = st.session_state.last_info or {}
    if last_info.get("diagnostic_result") is not None:
        st.markdown("**Actual environment diagnostic result**")
        st.json(last_info["diagnostic_result"])
    if last_info.get("recovery_result") is not None:
        st.markdown("**Actual environment recovery result**")
        st.json(last_info["recovery_result"])

st.subheader("Episode action log")
if st.session_state.history:
    st.dataframe(pd.DataFrame(st.session_state.history), hide_index=True, use_container_width=True)
else:
    st.caption("Actions will appear here after the first decision.")

if st.session_state.done:
    info = st.session_state.last_info or {}
    if info.get("verified_recovery"):
        st.success("Recovery verified after two consecutive healthy checks.")
    elif info.get("safe_termination"):
        st.warning("Episode ended by safe shutdown; service was not restored.")
    elif info.get("failure"):
        st.error("Episode ended because the system reached a critical failure condition.")
    else:
        st.info("Episode ended or reached its time limit. Inspect the action log for the outcome.")

    reveal = st.toggle("Reveal hidden fault (after episode)", value=False)
    if reveal:
        st.info(f"Environment ground truth: **{fault_label(st.session_state.hidden_fault)}**")
    st.caption("The hidden fault label is ground truth and is not passed directly to the PPO policy.")
