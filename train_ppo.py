
import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

from environment.rl_environment import FaultDiagnosisEnv


# --------------------------------------------------
# V4 CONFIGURATION
# --------------------------------------------------
TOTAL_TIMESTEPS = 100_000
SEED = 2026

MODEL_PATH = "ppo_fault_diagnosis_v4"
LOG_DIR = "./ppo_logs_v4"
TENSORBOARD_DIR = "./tensorboard_logs_v4"


def make_env():
    """Create a monitored v4 environment."""
    env = FaultDiagnosisEnv(seed=SEED)
    return Monitor(env, filename=os.path.join(LOG_DIR, "training"))


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_DIR, exist_ok=True)

    print("=" * 60)
    print("PPO FAULT DIAGNOSIS - V4 TRAINING")
    print("=" * 60)
    print(f"Timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"Seed: {SEED}")
    print("Device: CPU")
    print(f"Model output: {MODEL_PATH}.zip")

    # Check the unwrapped Gymnasium environment before training.
    check_env(FaultDiagnosisEnv(seed=SEED), warn=True)
    print("\nEnvironment check passed.")

    env = make_env()

    try:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.02,
            n_epochs=10,
            seed=SEED,
            verbose=1,
            device="cpu",
            tensorboard_log=TENSORBOARD_DIR,
        )

        print("\nStarting PPO training...")
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            progress_bar=False,
            tb_log_name="PPO_v4",
        )

        model.save(MODEL_PATH)

        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)
        print(f"Model saved to: {MODEL_PATH}.zip")
        print("Previous model files were not overwritten.")
        print("\nNext command:")
        print("python evaluate_ppo.py")

    finally:
        env.close()


if __name__ == "__main__":
    main()
