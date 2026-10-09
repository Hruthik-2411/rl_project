
from environment.rl_environment import FaultDiagnosisEnv

env = FaultDiagnosisEnv()

def critical_conditions(state):
    return {
        "availability_below_20": state.availability < 20,
        "error_rate_above_80": state.error_rate > 80,
        "cpu_above_98": state.cpu > 98,
        "memory_above_98": state.memory > 98,
        "network_latency_above_400": state.network_latency > 400,
        "db_failure_rate_above_80": state.db_failure_rate > 80,
    }

metric_names = [
    "cpu", "memory", "response_time", "error_rate",
    "queue_length", "availability", "network_latency",
    "packet_loss", "db_latency", "db_failure_rate",
]

for action in [0, 3, 4, 7, 9]:
    obs, reset_info = env.reset(seed=42)
    before = env.system.get_state()

    before_metrics = {
        key: float(getattr(before, key))
        for key in metric_names
    }

    obs, reward, terminated, truncated, info = env.step(action)
    after = env.system.get_state()

    after_metrics = {
        key: float(getattr(after, key))
        for key in metric_names
    }

    print(f"\n{'=' * 55}")
    print(f"Action: {action} | Fault: {reset_info['hidden_fault']}")
    print(f"Reward: {reward:.2f} | Terminated: {terminated}")
    print("BEFORE:", before_metrics)
    print("AFTER: ", after_metrics)
    print("Critical conditions:", critical_conditions(after))
    print("Failure info:", info.get("failure", False))

env.close()
