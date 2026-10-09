
import csv
import os

import numpy as np
from stable_baselines3 import PPO

from environment.rl_environment import FaultDiagnosisEnv
from environment.faults import FaultType


# --------------------------------------------------
# V4 EVALUATION CONFIGURATION
# --------------------------------------------------
MODEL_PATH = "ppo_fault_diagnosis_v4.zip"
EPISODES_PER_FAULT = 40
BASE_SEED = 50_000
RESULTS_PATH = "evaluation_v4.csv"


def evaluate_episode(model, fault_type, seed):
    """Evaluate one episode with a specified hidden fault."""

    env = FaultDiagnosisEnv(seed=seed)

    try:
        observation, info = env.reset(seed=seed)

        # Controlled test setup: ensure balanced fault coverage.
        # The hidden fault is never added to the agent observation.
        severity = 0.75
        env.hidden_fault = fault_type
        env.hidden_severity = severity
        env.fault_system.inject_fault(fault_type, severity)

        observation = env._get_observation()

        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        diagnostic_attempts = 0
        available_queries = 0
        reliable_queries = 0

        verified_recovery = False
        safe_termination = False
        critical_failure = False

        final_info = info

        while not terminated and not truncated:
            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            observation, reward, terminated, truncated, info = (
                env.step(int(action))
            )

            total_reward += float(reward)
            steps += 1
            final_info = info

            diagnostic_result = info.get("diagnostic_result")

            if diagnostic_result is not None:
                diagnostic_attempts += 1

                if diagnostic_result.get("available", False):
                    available_queries += 1

                    if diagnostic_result.get("reliable", False):
                        reliable_queries += 1

            if info.get("verified_recovery", False):
                verified_recovery = True

            if info.get("safe_termination", False):
                safe_termination = True

            if info.get("failure", False):
                critical_failure = True

        # The belief is the agent's estimate, not the hidden fault.
        final_belief = np.asarray(
            final_info.get("belief", env.belief),
            dtype=np.float64,
        )

        fault_names = [fault.name for fault in FaultType]
        predicted_fault = fault_names[int(np.argmax(final_belief))]
        diagnosis_correct = predicted_fault == fault_type.name

        query_count = int(final_info.get("query_count", env.query_count))
        accumulated_cost = float(
            final_info.get("accumulated_cost", env.accumulated_cost)
        )

        return {
            "fault": fault_type.name,
            "reward": total_reward,
            "steps": steps,
            "query_count": query_count,
            "diagnostic_attempts": diagnostic_attempts,
            "available_queries": available_queries,
            "reliable_queries": reliable_queries,
            "diagnostic_cost": accumulated_cost,
            "predicted_fault": predicted_fault,
            "diagnosis_correct": diagnosis_correct,
            "verified_recovery": verified_recovery,
            "safe_termination": safe_termination,
            "critical_failure": critical_failure,
            "remaining_budget": float(env.remaining_budget),
        }

    finally:
        env.close()


def mean_metric(rows, key):
    if not rows:
        return 0.0
    return float(np.mean([row[key] for row in rows]))


def print_summary(title, rows):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    count = len(rows)
    if count == 0:
        print("No episodes to evaluate.")
        return

    diagnosed = [row for row in rows if row["query_count"] > 0]

    print(f"Episodes: {count}")
    print(f"Mean reward: {mean_metric(rows, 'reward'):.2f}")
    print(f"Mean steps: {mean_metric(rows, 'steps'):.2f}")
    print(f"Mean diagnostic queries: {mean_metric(rows, 'query_count'):.2f}")
    print(f"Mean diagnostic attempts: {mean_metric(rows, 'diagnostic_attempts'):.2f}")
    print(f"Mean available queries: {mean_metric(rows, 'available_queries'):.2f}")
    print(f"Mean reliable queries: {mean_metric(rows, 'reliable_queries'):.2f}")
    print(f"Mean diagnostic cost: {mean_metric(rows, 'diagnostic_cost'):.2f}")
    print(f"Mean remaining budget: {mean_metric(rows, 'remaining_budget'):.2f}")

    print(
        f"Verified recovery rate: "
        f"{100 * mean_metric(rows, 'verified_recovery'):.1f}%"
    )
    print(
        f"Safe termination rate: "
        f"{100 * mean_metric(rows, 'safe_termination'):.1f}%"
    )
    print(
        f"Critical failure rate: "
        f"{100 * mean_metric(rows, 'critical_failure'):.1f}%"
    )

    print(
        f"Belief top-1 accuracy, all episodes: "
        f"{100 * mean_metric(rows, 'diagnosis_correct'):.1f}%"
    )

    if diagnosed:
        print(
            f"Belief top-1 accuracy, episodes with queries: "
            f"{100 * mean_metric(diagnosed, 'diagnosis_correct'):.1f}%"
        )
        print(
            f"Diagnostic coverage: {100 * len(diagnosed) / count:.1f}%"
        )
    else:
        print("Diagnostic coverage: 0.0%")

    print(
        "Note: top-1 accuracy measures the final belief distribution; "
        "it does not necessarily mean the agent explicitly committed "
        "to a diagnosis before recovering."
    )


def main():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run python train_ppo.py first."
        )

    print("Loading:", MODEL_PATH)
    model = PPO.load(MODEL_PATH, device="cpu")

    all_results = []

    fault_types = list(FaultType)

    print(f"\nFault types: {len(fault_types)}")
    print(f"Episodes per fault: {EPISODES_PER_FAULT}")
    print(f"Total episodes: {len(fault_types) * EPISODES_PER_FAULT}")

    for fault_index, fault_type in enumerate(fault_types):
        fault_results = []

        print(f"\nEvaluating {fault_type.name}...")

        for episode_index in range(EPISODES_PER_FAULT):
            seed = (
                BASE_SEED
                + fault_index * EPISODES_PER_FAULT
                + episode_index
            )

            result = evaluate_episode(
                model,
                fault_type,
                seed,
            )

            fault_results.append(result)
            all_results.append(result)

        print_summary(f"RESULTS: {fault_type.name}", fault_results)

    print_summary("OVERALL V4 EVALUATION", all_results)

    fieldnames = list(all_results[0].keys())

    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nPer-episode results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
