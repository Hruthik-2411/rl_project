"""Evaluate the v2 PPO model over independently seeded episodes."""
import numpy as np
from stable_baselines3 import PPO
from environment.rl_environment import FaultDiagnosisEnv

N_EPISODES = 100
MODEL_PATH = "ppo_fault_diagnosis_v2.zip"

def main():
    model = PPO.load(MODEL_PATH, device="cpu")
    rows = []
    for seed in range(10_000, 10_000 + N_EPISODES):
        env = FaultDiagnosisEnv()
        obs, reset_info = env.reset(seed=seed)
        total_reward = 0.0
        steps = 0
        done = False
        final_info = {}
        while not done and steps < env.max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(np.asarray(action).item()))
            total_reward += float(reward)
            steps += 1
            done = bool(terminated or truncated)
            final_info = info
        rows.append({
            "seed": seed,
            "fault": reset_info.get("hidden_fault", "unknown"),
            "reward": total_reward,
            "steps": steps,
            "verified_recovery": bool(final_info.get("verified_recovery", False)),
            "safe_termination": bool(final_info.get("safe_termination", False)),
            "critical_failure": bool(final_info.get("failure", False)),
            "diagnostic_cost": float(env.accumulated_cost),
            "queries": int(env.query_count),
            "elapsed_time": int(env.elapsed_time),
        })
        env.close()

    print(f"Episodes: {len(rows)}")
    print(f"Mean reward: {np.mean([r['reward'] for r in rows]):.2f}")
    print(f"Mean steps: {np.mean([r['steps'] for r in rows]):.2f}")
    print(f"Verified recovery rate: {100*np.mean([r['verified_recovery'] for r in rows]):.1f}%")
    print(f"Safe termination rate: {100*np.mean([r['safe_termination'] for r in rows]):.1f}%")
    print(f"Critical failure rate: {100*np.mean([r['critical_failure'] for r in rows]):.1f}%")
    print(f"Mean diagnostic cost: {np.mean([r['diagnostic_cost'] for r in rows]):.2f}")
    print(f"Mean diagnostic queries: {np.mean([r['queries'] for r in rows]):.2f}")
    print(f"Mean elapsed time: {np.mean([r['elapsed_time'] for r in rows]):.2f}")
    print("\\nPer-fault verified recovery:")
    for fault in sorted(set(r["fault"] for r in rows)):
        subset = [r for r in rows if r["fault"] == fault]
        rate = 100*np.mean([r["verified_recovery"] for r in subset])
        print(f"  {fault}: {rate:.1f}% ({len(subset)} episodes)")
    import csv
    with open("ppo_v2_evaluation.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("\\nDetailed results saved to ppo_v2_evaluation.csv")

if __name__ == "__main__":
    main()
