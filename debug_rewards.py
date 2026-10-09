
from environment.rl_environment import FaultDiagnosisEnv

env = FaultDiagnosisEnv()

for action in range(11):
    obs, reset_info = env.reset(seed=42)

    obs, reward, terminated, truncated, info = env.step(action)

    print(f"\nAction: {action}")
    print(f"Hidden fault: {reset_info['hidden_fault']}")
    print(f"Reward: {reward:.2f}")
    print(f"Terminated: {terminated}")

    if "diagnostic_result" in info:
        print("Diagnostic result:", info["diagnostic_result"])

    if "recovery_result" in info:
        print("Recovery result:", info["recovery_result"])

    print("Failure:", info.get("failure", False))
    print("Information gain:", info.get("information_gain", 0.0))

env.close()
