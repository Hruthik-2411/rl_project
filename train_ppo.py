"""Train a fresh PPO policy for the recovery-verification environment.

Run from the project root:
    python train_ppo.py
"""
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from environment.rl_environment import FaultDiagnosisEnv

TOTAL_TIMESTEPS = 200_000

def main():
    env = FaultDiagnosisEnv()
    check_env(env, warn=True)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.02,
        n_epochs=10,
        device="auto",
        seed=2026,
        tensorboard_log="./ppo_tensorboard/",
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
    model.save("ppo_fault_diagnosis_v2")
    env.close()
    print("Saved new model: ppo_fault_diagnosis_v2.zip")
    print("Use this newly trained model with the updated Streamlit app.")

if __name__ == "__main__":
    main()
