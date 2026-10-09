
from environment.rl_environment import FaultDiagnosisEnv

env = FaultDiagnosisEnv()

print("Testing all 11 actions...\n")

for action in range(11):
    try:
        obs, info = env.reset(seed=42)

        obs, reward, terminated, truncated, info = env.step(action)

        valid = env.observation_space.contains(obs)

        print(
            f"Action {action}: "
            f"reward={reward:.2f}, "
            f"shape={obs.shape}, "
            f"valid={valid}, "
            f"terminated={terminated}, "
            f"truncated={truncated}"
        )

    except Exception as exc:
        print(
            f"Action {action} FAILED: "
            f"{type(exc).__name__}: {exc}"
        )

env.close()
