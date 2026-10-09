
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from environment.system_model import DistributedSystem
from environment.faults import FaultDynamics, FaultType
from environment.diagnostics import DiagnosticManager, DiagnosticSource
from environment.recovery import RecoveryManager, RecoveryAction


class FaultDiagnosisEnv(gym.Env):
    """
    Cost-aware POMDP environment for fault diagnosis and recovery.

    Observation:
        11 normalized system metrics
        5 fault-belief probabilities
        7 additional state variables
        Total: 23

    Actions:
        0-3: Diagnostics
        4-9: Recovery interventions
        10: Verification / wait

    The hidden fault is not included in the observation.
    """

    metadata = {"render_modes": []}

    FAULT_COUNT = 5
    OBSERVATION_SIZE = 23
    INITIAL_BUDGET = 10.0

    def __init__(self, seed=None):
        super().__init__()

        self.seed_value = seed
        self.rng = np.random.default_rng(seed)

        self.system = DistributedSystem(seed=seed)
        self.fault_system = FaultDynamics(self.system, seed=seed)
        self.diagnostics = DiagnosticManager(seed=seed)
        self.recovery = RecoveryManager(
            self.system, self.fault_system, seed=seed
        )

        # 0-3 diagnostics, 4-9 recovery, 10 verification.
        self.action_space = spaces.Discrete(11)

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.OBSERVATION_SIZE,),
            dtype=np.float32,
        )

        self.initial_budget = self.INITIAL_BUDGET
        self.max_steps = 30

        self._reset_episode_variables()

    def _reset_episode_variables(self):
        self.remaining_budget = self.initial_budget
        self.accumulated_cost = 0.0
        self.query_count = 0
        self.elapsed_time = 0
        self.step_count = 0

        self.pending_verification = False
        self.healthy_verification_streak = 0
        self.recovery_attempts = 0

        self.belief = np.ones(
            self.FAULT_COUNT, dtype=np.float32
        ) / self.FAULT_COUNT

        self.hidden_fault = None
        self.hidden_severity = None

        self.episode_verified_recovery = False
        self.episode_safe_termination = False
        self.episode_failure = False

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed_value = seed
            self.rng = np.random.default_rng(seed)

            self.system = DistributedSystem(seed=seed)
            self.fault_system = FaultDynamics(
                self.system, seed=seed
            )
            self.diagnostics = DiagnosticManager(seed=seed)
            self.recovery = RecoveryManager(
                self.system, self.fault_system, seed=seed
            )

        super().reset(seed=seed)
        self.system.reset()
        self._reset_episode_variables()

        fault_type = self.rng.choice(list(FaultType))
        severity = float(self.rng.uniform(0.5, 1.0))

        self.hidden_fault = fault_type
        self.hidden_severity = severity

        self.fault_system.inject_fault(fault_type, severity)

        observation = self._get_observation()

        # Hidden fault is returned in info for evaluation/debugging,
        # never included in the agent's observation.
        info = {
            "hidden_fault": fault_type.name,
            "severity": severity,
        }

        return observation, info

    def step(self, action):
        action = int(action)

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        self.step_count += 1
        reward = -0.5  # Small cost for every decision.
        terminated = False
        truncated = False
        info = {}

        # --------------------------------------------------
        # 0-3: DIAGNOSTIC ACTIONS
        # --------------------------------------------------
        if self.pending_verification and action != 10:
            # Require verification immediately after intervention.
            reward -= 3.0
            info["action_deferred"] = (
                "Recovery verification is pending; choose action 10."
            )

        elif action < 4:
            source = DiagnosticSource(action)
            config = self.diagnostics.sources[source]

            cost = float(config["cost"])
            delay = int(config["delay"])

            if cost > self.remaining_budget:
                reward -= 5.0
                info["budget_violation"] = True
                info["diagnostic_result"] = {
                    "source": source.name,
                    "available": False,
                    "evidence": {},
                    "confidence": 0.0,
                    "cost": cost,
                    "delay": 0,
                    "reliable": False,
                    "reason": "INSUFFICIENT_BUDGET",
                    "information_gain": 0.0,
                }
            else:
                uncertainty_before = self._calculate_uncertainty()

                # Account for diagnostic latency.
                for _ in range(delay):
                    self.system.step_time()
                    self.fault_system.apply_fault()

                result = self.diagnostics.query(
                    source, self.fault_system.fault
                )

                self.remaining_budget = max(
                    0.0, self.remaining_budget - cost
                )
                self.accumulated_cost += cost
                self.query_count += 1
                self.elapsed_time += delay

                # More moderate cost and latency penalties.
                reward -= 0.5 * cost
                reward -= 0.25 * delay

                self._update_belief(result)

                uncertainty_after = self._calculate_uncertainty()
                information_gain = max(
                    0.0, uncertainty_before - uncertainty_after
                )

                available = result.get("available", True)
                reliable = result.get("reliable", False)

                if available and reliable:
                    reward += 12.0 * information_gain

                # Additional reward for useful, high-confidence
                # fault identification from a diagnostic source.
                evidence = result.get("evidence", {})
                identified = evidence.get("identified_fault")

                if available and reliable and identified:
                    if identified == self.hidden_fault.name:
                        reward += 5.0
                        info["diagnosis_correct"] = True
                    else:
                        reward -= 2.0
                        info["diagnosis_correct"] = False

                result["uncertainty_before"] = float(
                    uncertainty_before
                )
                result["uncertainty_after"] = float(
                    uncertainty_after
                )
                result["information_gain"] = float(
                    information_gain
                )

                info["diagnostic_result"] = result
                info["information_gain"] = float(information_gain)

        # --------------------------------------------------
        # 4-9: RECOVERY ACTIONS
        # --------------------------------------------------
        elif action < 10:
            recovery_action = RecoveryAction(action - 4)
            result = self.recovery.execute(recovery_action)

            info["recovery_result"] = result

            recovery_cost = float(result.get("cost", 0.0))
            recovery_delay = int(result.get("delay", 0))

            self.elapsed_time += recovery_delay
            self.recovery_attempts += 1

            # Recovery cost still matters, but does not dominate
            # all the diagnostic rewards.
            reward -= 0.5 * recovery_cost
            reward -= 0.25 * recovery_delay

            if result.get("safe_termination", False):
                # Stopping can be appropriate, but is not equivalent
                # to restoring the service.
                reward -= 12.0
                terminated = True

                self.pending_verification = False
                self.episode_safe_termination = True

                info["safe_termination"] = True
                info["recovery_success"] = False

            elif result.get("success", False):
                # The intervention is not yet a verified recovery.
                reward += 12.25

                self.pending_verification = True
                self.healthy_verification_streak = 0

                info["safe_termination"] = False
                info["recovery_success"] = False
                info["verification_required"] = True

            else:
                # Penalize a failed intervention without making the
                # agent's only sensible strategy to avoid diagnosis.
                reward -= 12.0

                self.pending_verification = False
                self.healthy_verification_streak = 0

                info["safe_termination"] = False
                info["recovery_success"] = False
                info["intervention_failed"] = True

        # --------------------------------------------------
        # 10: VERIFICATION / WAIT
        # --------------------------------------------------
        else:
            if self.pending_verification:
                self.elapsed_time += 1

                healthy = self._is_system_healthy()

                info["verification_performed"] = True
                info["system_healthy_this_check"] = bool(healthy)

                if healthy:
                    self.healthy_verification_streak += 1
                    reward += 5.0
                else:
                    self.healthy_verification_streak = 0
                    self.pending_verification = False
                    reward -= 4.0

                if self.healthy_verification_streak >= 2:
                    # Credit recovery only after two healthy checks.
                    reward += 40.0
                    terminated = True

                    self.pending_verification = False
                    self.fault_system.clear_fault()

                    self.episode_verified_recovery = True

                    info["recovery_success"] = True
                    info["verified_recovery"] = True
                else:
                    info["verified_recovery"] = False

            else:
                # Waiting without a pending verification wastes time.
                self.elapsed_time += 1
                reward -= 2.0

                info["verification_performed"] = False
                info["verified_recovery"] = False

        # --------------------------------------------------
        # ADVANCE THE SYSTEM
        # --------------------------------------------------
        if not terminated:
            self.system.step_time()

            # Fault progression pauses while the agent verifies an
            # intervention. Otherwise the fault continues to evolve.
            if (
                not self.pending_verification
                and not info.get("verified_recovery", False)
            ):
                self.fault_system.apply_fault()

            self.elapsed_time += 1
            reward -= 0.5

        # --------------------------------------------------
        # CRITICAL-FAILURE CHECK
        # --------------------------------------------------
        state = self.system.get_state()

        if not info.get("safe_termination", False):
            critical_failure = (
                state.availability < 20
                or state.error_rate > 80
                or state.cpu > 98
                or state.memory > 98
                or state.network_latency > 400
                or state.db_failure_rate > 80
            )

            if critical_failure:
                reward -= 50.0
                terminated = True
                self.pending_verification = False
                self.episode_failure = True

                info["failure"] = True
            else:
                info["failure"] = False

        # --------------------------------------------------
        # TIME LIMIT
        # --------------------------------------------------
        if self.step_count >= self.max_steps and not terminated:
            truncated = True
            info["time_limit_reached"] = True

        observation = self._get_observation()

        info.update({
            "remaining_budget": float(self.remaining_budget),
            "accumulated_cost": float(self.accumulated_cost),
            "query_count": int(self.query_count),
            "step_count": int(self.step_count),
            "elapsed_time": int(self.elapsed_time),
            "belief": self.belief.copy(),
            "uncertainty": float(self._calculate_uncertainty()),
            "pending_verification": bool(self.pending_verification),
            "healthy_verification_streak": int(
                self.healthy_verification_streak
            ),
            "hidden_fault": (
                self.hidden_fault.name if self.hidden_fault else None
            ),
            "hidden_severity": float(self.fault_system.severity),
            "verified_recovery": bool(
                info.get("verified_recovery", False)
            ),
        })

        return observation, float(reward), terminated, truncated, info

    # ======================================================
    # HEALTH CHECK
    # ======================================================
    def _is_system_healthy(self):
        s = self.system.get_state()

        return bool(
            s.cpu < 85
            and s.memory < 88
            and s.response_time < 150
            and s.error_rate < 10
            and s.queue_length < 100
            and s.availability >= 95
            and s.network_latency < 180
            and s.packet_loss < 5
            and s.db_latency < 160
            and s.db_failure_rate < 10
        )

    # ======================================================
    # BELIEF UPDATE
    # ======================================================
    def _update_belief(self, diagnostic_result):
        if not diagnostic_result.get("available", True):
            return

        evidence = diagnostic_result.get("evidence", {})
        reliable = diagnostic_result.get("reliable", False)
        confidence = float(
            np.clip(
                diagnostic_result.get("confidence", 0.0),
                0.0,
                1.0,
            )
        )

        fault_list = list(FaultType)
        scores = np.ones(len(fault_list), dtype=np.float64)

        # Direct identification from deep diagnostics.
        identified_fault = evidence.get("identified_fault")

        if identified_fault:
            matching_indices = [
                i for i, fault in enumerate(fault_list)
                if fault.name == identified_fault
            ]

            if matching_indices:
                idx = matching_indices[0]
                strength = (
                    max(0.75, confidence)
                    if reliable
                    else 0.55
                )

                posterior = np.full(
                    len(fault_list),
                    (1.0 - strength) / (len(fault_list) - 1),
                    dtype=np.float64,
                )
                posterior[idx] = strength

                self.belief = posterior.astype(np.float32)
                self.belief /= self.belief.sum()
                return

        # Evidence generated by telemetry.
        for i, fault in enumerate(fault_list):
            name = fault.name

            if name == "CPU_OVERLOAD":
                if evidence.get("cpu_high", False):
                    scores[i] *= 4.0
                if evidence.get("response_time_high", False):
                    scores[i] *= 1.5
                if evidence.get("error_rate_high", False):
                    scores[i] *= 1.3

            elif name == "MEMORY_LEAK":
                if evidence.get("memory_high", False):
                    scores[i] *= 4.0
                if evidence.get("response_time_high", False):
                    scores[i] *= 1.5

            elif name == "NETWORK_DEGRADATION":
                if evidence.get("network_latency_high", False):
                    scores[i] *= 4.0
                if evidence.get("packet_loss_high", False):
                    scores[i] *= 4.0

            elif name == "DATABASE_FAILURE":
                if evidence.get("db_latency_high", False):
                    scores[i] *= 4.0
                if evidence.get("db_failure_high", False):
                    scores[i] *= 4.0

            elif name == "SERVICE_FAILURE":
                if evidence.get("availability_low", False):
                    scores[i] *= 4.0
                if evidence.get("error_rate_high", False):
                    scores[i] *= 2.0

            # Health probe evidence.
            application = evidence.get("application")
            network = evidence.get("network")
            database = evidence.get("database")

            if (
                name in {
                    "CPU_OVERLOAD",
                    "MEMORY_LEAK",
                    "SERVICE_FAILURE",
                }
                and application in {"DEGRADED", "FAILED"}
            ):
                scores[i] *= 2.0

            if (
                name == "NETWORK_DEGRADATION"
                and network in {"DEGRADED", "FAILING"}
            ):
                scores[i] *= 4.0

            if (
                name == "DATABASE_FAILURE"
                and database in {"DEGRADED", "FAILING"}
            ):
                scores[i] *= 4.0

        # Log messages may provide useful textual clues.
        events = evidence.get("events", [])
        if isinstance(events, str):
            events = [events]

        event_text = " ".join(str(event) for event in events).lower()

        keyword_groups = {
            "CPU_OVERLOAD": ("cpu", "saturation", "overload"),
            "MEMORY_LEAK": ("memory leak", "memory growth", "heap"),
            "NETWORK_DEGRADATION": (
                "network", "packet loss", "latency", "connection timeout"
            ),
            "DATABASE_FAILURE": (
                "database", "db failure", "query timeout", "connection pool"
            ),
            "SERVICE_FAILURE": (
                "service down", "service failure", "crash", "unavailable"
            ),
        }

        for i, fault in enumerate(fault_list):
            if any(
                keyword in event_text
                for keyword in keyword_groups.get(fault.name, ())
            ):
                scores[i] *= 3.0

        # Bayesian-style evidence update.
        prior = np.maximum(self.belief.astype(np.float64), 1e-8)
        posterior = prior * scores
        total = posterior.sum()

        if total <= 0 or not np.isfinite(total):
            return

        posterior /= total

        if reliable:
            strength = float(np.clip(confidence, 0.45, 0.90))
        else:
            strength = 0.20

        updated = (1.0 - strength) * prior + strength * posterior
        updated /= updated.sum()

        self.belief = updated.astype(np.float32)

    # ======================================================
    # UNCERTAINTY
    # ======================================================
    def _calculate_uncertainty(self):
        belief = np.clip(self.belief, 1e-10, 1.0)
        entropy = -np.sum(belief * np.log(belief))
        return float(np.clip(entropy / np.log(self.FAULT_COUNT), 0.0, 1.0))

    # ======================================================
    # OBSERVATION
    # ======================================================
    def _get_observation(self):
        state = self.system.get_state()

        system_values = np.array([
            state.cpu / 100.0,
            state.memory / 100.0,
            state.request_rate / 200.0,
            state.response_time / 500.0,
            state.error_rate / 100.0,
            state.queue_length / 200.0,
            state.availability / 100.0,
            state.network_latency / 500.0,
            state.packet_loss / 100.0,
            state.db_latency / 500.0,
            state.db_failure_rate / 100.0,
        ], dtype=np.float32)

        uncertainty = self._calculate_uncertainty()

        extra_values = np.array([
            uncertainty,
            self.remaining_budget / self.initial_budget,
            min(self.accumulated_cost / 20.0, 1.0),
            min(self.query_count / 10.0, 1.0),
            min(self.elapsed_time / 60.0, 1.0),
            float(self.pending_verification),
            min(self.healthy_verification_streak / 2.0, 1.0),
        ], dtype=np.float32)

        observation = np.concatenate([
            system_values,
            self.belief,
            extra_values,
        ])

        return np.clip(
            observation, 0.0, 1.0
        ).astype(np.float32)


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    env = FaultDiagnosisEnv(seed=42)
    check_env(env, skip_render_check=True)

    observation, info = env.reset(seed=42)

    print("Environment check passed.")
    print("Hidden fault (debug only):", info["hidden_fault"])
    print("Observation shape:", observation.shape)
    print("Initial belief:", observation[11:16])
    print("Initial uncertainty:", observation[16])

    for action in range(11):
        test_env = FaultDiagnosisEnv(seed=42)
        test_env.reset(seed=42)
        result = test_env.step(action)
        print(
            f"Action {action}: reward={result[1]:.2f}, "
            f"terminated={result[2]}, truncated={result[3]}"
        )
