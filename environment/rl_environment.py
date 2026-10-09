import gymnasium as gym
from gymnasium import spaces
import numpy as np

from environment.system_model import DistributedSystem
from environment.faults import FaultDynamics, FaultType
from environment.diagnostics import DiagnosticManager, DiagnosticSource
from environment.recovery import RecoveryManager, RecoveryAction


class FaultDiagnosisEnv(gym.Env):
    """
    POMDP environment for cost-aware active fault diagnosis
    and recovery in a distributed IT system.

    The hidden fault is never directly provided to the PPO agent.
    """

    metadata = {"render_modes": []}

    def __init__(self, seed=None):
        super().__init__()

        self.seed_value = seed
        self.rng = np.random.default_rng(seed)

        # --------------------------------------------------
        # SYSTEM COMPONENTS
        # --------------------------------------------------

        self.system = DistributedSystem(seed=seed)

        self.fault_system = FaultDynamics(
            self.system,
            seed=seed
        )

        self.diagnostics = DiagnosticManager(
            seed=seed
        )

        self.recovery = RecoveryManager(
            self.system,
            self.fault_system,
            seed=seed
        )

        # --------------------------------------------------
        # ACTION SPACE
        # --------------------------------------------------
        #
        # 0 -> TELEMETRY
        # 1 -> HEALTH_PROBE
        # 2 -> LOGS
        # 3 -> DEEP_DIAGNOSTIC
        #
        # 4 -> RESTART_SERVICE
        # 5 -> REROUTE_TRAFFIC
        # 6 -> ISOLATE_COMPONENT
        # 7 -> REDUCE_LOAD
        # 8 -> ROLLBACK
        # 9 -> STOP_SYSTEM
        #
        self.action_space = spaces.Discrete(11)  # 0-3 diagnostics, 4-9 recovery, 10 verify/wait

        # --------------------------------------------------
        # OBSERVATION SPACE
        # --------------------------------------------------
        #
        # 11 system metrics
        # 5 fault beliefs
        # 1 uncertainty
        # 1 remaining diagnostic budget
        # 1 accumulated diagnostic cost
        # 1 query count
        # 1 elapsed time
        #
        # 1 pending recovery verification
        # 1 consecutive healthy verification count
        # Total = 23
        #
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(23,),
            dtype=np.float32
        )

        # --------------------------------------------------
        # EPISODE PARAMETERS
        # --------------------------------------------------

        self.initial_budget = 10.0
        self.remaining_budget = self.initial_budget

        self.accumulated_cost = 0.0
        self.query_count = 0

        self.elapsed_time = 0
        self.step_count = 0
        self.pending_verification = False
        self.healthy_verification_streak = 0
        self.recovery_attempts = 0

        self.max_steps = 30

        # Initial uniform belief over 5 faults.
        self.belief = (
            np.ones(5, dtype=np.float32) / 5.0
        )

        # Hidden variables.
        # These are NOT included in observations.
        self.hidden_fault = None
        self.hidden_severity = None

    # ======================================================
    # RESET
    # ======================================================

    def reset(self, seed=None, options=None):

        if seed is not None:

            self.seed_value = seed

            self.rng = np.random.default_rng(seed)

            self.system = DistributedSystem(
                seed=seed
            )

            self.fault_system = FaultDynamics(
                self.system,
                seed=seed
            )

            self.diagnostics = DiagnosticManager(
                seed=seed
            )

            self.recovery = RecoveryManager(
                self.system,
                self.fault_system,
                seed=seed
            )

        super().reset(seed=seed)

        # Reset distributed system.
        self.system.reset()

        # Reset episode variables.
        self.remaining_budget = self.initial_budget

        self.accumulated_cost = 0.0

        self.query_count = 0

        self.elapsed_time = 0

        self.step_count = 0
        self.pending_verification = False
        self.healthy_verification_streak = 0
        self.recovery_attempts = 0

        # Reset belief.
        self.belief = (
            np.ones(5, dtype=np.float32) / 5.0
        )

        # --------------------------------------------------
        # GENERATE HIDDEN FAULT
        # --------------------------------------------------

        fault_type = self.rng.choice(
            list(FaultType)
        )

        severity = self.rng.uniform(
            0.5,
            1.0
        )

        self.hidden_fault = fault_type

        self.hidden_severity = severity

        self.fault_system.inject_fault(
            fault_type,
            severity
        )

        observation = self._get_observation()

        info = {
            "hidden_fault": fault_type.name,
            "severity": float(severity)
        }

        return observation, info

    # ======================================================
    # STEP
    # ======================================================

    def step(self, action):
        """One decision step in the diagnosis/recovery process."""
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        self.step_count += 1
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        # Once an intervention is awaiting verification, reserve the next
        # decision for monitoring. If that check fails, diagnosis can resume.
        if self.pending_verification and action != 10:
            reward -= 8.0
            info["action_deferred"] = "Recovery verification is pending; choose VERIFY_RECOVERY."
        # 0-3: acquire diagnostic information
        elif action < 4:
            source = DiagnosticSource(action)
            config = self.diagnostics.sources[source]
            cost = float(config["cost"])
            delay = int(config["delay"])

            if cost > self.remaining_budget:
                reward -= 10.0
                info["diagnostic_result"] = {
                    "source": source.name, "available": False,
                    "evidence": {}, "confidence": 0.0,
                    "cost": cost, "delay": 0, "reliable": False,
                    "reason": "INSUFFICIENT_BUDGET"
                }
                info["budget_violation"] = True
            else:
                # The source's latency is real simulated time: the system and
                # active fault continue evolving while the query runs.
                for _ in range(delay):
                    self.system.step_time()
                    self.fault_system.apply_fault()
                result = self.diagnostics.query(source, self.fault_system.fault)
                self.remaining_budget = max(0.0, self.remaining_budget - cost)
                self.accumulated_cost += cost
                self.query_count += 1
                self.elapsed_time += delay
                reward -= cost + 2.0 * delay
                self._update_belief(result)
                info["diagnostic_result"] = result

            if self.pending_verification:
                reward -= 1.0

        # 4-9: recovery actions
        elif action < 10:
            recovery_action = RecoveryAction(action - 4)
            result = self.recovery.execute(recovery_action)
            info["recovery_result"] = result
            recovery_cost = float(result.get("cost", 0.0))
            recovery_delay = int(result.get("delay", 0))
            self.elapsed_time += recovery_delay
            reward -= recovery_cost + 2.0 * recovery_delay
            self.recovery_attempts += 1

            if result.get("safe_termination", False):
                reward -= 10.0
                terminated = True
                info["safe_termination"] = True
                info["recovery_success"] = False
                self.pending_verification = False
            elif result.get("success", False):
                # The intervention worked internally; service restoration
                # is not credited until consecutive monitoring checks pass.
                reward += 8.0
                self.pending_verification = True
                self.healthy_verification_streak = 0
                info["safe_termination"] = False
                info["recovery_success"] = False
                info["verification_required"] = True
            else:
                reward -= 30.0
                self.pending_verification = False
                self.healthy_verification_streak = 0
                info["safe_termination"] = False
                info["recovery_success"] = False

        # 10: explicit verification / wait action
        else:
            if self.pending_verification:
                self.elapsed_time += 1
                healthy = self._is_system_healthy()
                info["verification_performed"] = True
                info["system_healthy_this_check"] = bool(healthy)
                if healthy:
                    self.healthy_verification_streak += 1
                    reward += 4.0
                else:
                    self.healthy_verification_streak = 0
                    self.pending_verification = False
                    reward -= 4.0

                if self.healthy_verification_streak >= 2:
                    reward += 100.0
                    terminated = True
                    self.pending_verification = False
                    self.fault_system.clear_fault()
                    info["recovery_success"] = True
                    info["verified_recovery"] = True
                else:
                    info["verified_recovery"] = False
            else:
                self.elapsed_time += 1
                reward -= 3.0
                info["verification_performed"] = False
                info["verified_recovery"] = False

        # Advance the simulated system. A cleared fault stops progressing.
        if not terminated:
            self.system.step_time()
            # Pause fault progression during verification: the intervention
            # has been applied, and monitoring checks whether service stays healthy.
            if not self.pending_verification and not info.get("verified_recovery", False):
                self.fault_system.apply_fault()
            self.elapsed_time += 1
            reward -= 2.0

        state = self.system.get_state()
        if not info.get("safe_termination", False):
            if (
                state.availability < 20
                or state.error_rate > 80
                or state.cpu > 98
                or state.memory > 98
                or state.network_latency > 400
                or state.db_failure_rate > 80
            ):
                reward -= 100.0
                terminated = True
                info["failure"] = True
                self.pending_verification = False

        # Budget exhaustion blocks additional queries but does not block recovery.
        if self.step_count >= self.max_steps and not terminated:
            truncated = True
            info["time_limit_reached"] = True

        observation = self._get_observation()
        info["remaining_budget"] = float(self.remaining_budget)
        info["accumulated_cost"] = float(self.accumulated_cost)
        info["query_count"] = int(self.query_count)
        info["step_count"] = int(self.step_count)
        info["elapsed_time"] = int(self.elapsed_time)
        info["belief"] = self.belief.copy()
        info["uncertainty"] = float(self._calculate_uncertainty())
        info["pending_verification"] = bool(self.pending_verification)
        info["healthy_verification_streak"] = int(self.healthy_verification_streak)
        info["hidden_fault"] = self.hidden_fault.name if self.hidden_fault else None
        info["hidden_severity"] = float(self.fault_system.severity)
        return observation, float(reward), terminated, truncated, info

    def _is_system_healthy(self):
        """Health gate: require service-level metrics to be back in range."""
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

    def _update_belief(
        self,
        diagnostic_result
    ):

        evidence = diagnostic_result.get(
            "evidence",
            {}
        )

        confidence = diagnostic_result.get(
            "confidence",
            0.0
        )

        reliable = diagnostic_result.get(
            "reliable",
            False
        )

        available = diagnostic_result.get(
            "available",
            True
        )

        # --------------------------------------------------
        # SOURCE UNAVAILABLE
        # --------------------------------------------------

        if not available:

            updated = (
                0.95 * self.belief
                + 0.05 * (
                    np.ones(5) / 5.0
                )
            )

            updated /= np.sum(updated)

            self.belief = updated.astype(
                np.float32
            )

            return

        # --------------------------------------------------
        # EVIDENCE SCORES
        # --------------------------------------------------

        evidence_scores = np.ones(
            5,
            dtype=np.float64
        )

        for fault in FaultType:

            score = 1.0

            # ==============================================
            # CPU OVERLOAD
            # ==============================================

            if fault == FaultType.CPU_OVERLOAD:

                if evidence.get(
                    "cpu_high",
                    False
                ):
                    score *= 4.0

                if evidence.get(
                    "response_time_high",
                    False
                ):
                    score *= 1.5

                if evidence.get(
                    "error_rate_high",
                    False
                ):
                    score *= 1.3

            # ==============================================
            # MEMORY LEAK
            # ==============================================

            elif fault == FaultType.MEMORY_LEAK:

                if evidence.get(
                    "memory_high",
                    False
                ):
                    score *= 4.0

                if evidence.get(
                    "response_time_high",
                    False
                ):
                    score *= 1.4

            # ==============================================
            # NETWORK DEGRADATION
            # ==============================================

            elif fault == FaultType.NETWORK_DEGRADATION:

                if evidence.get(
                    "network_latency_high",
                    False
                ):
                    score *= 4.0

                if evidence.get(
                    "packet_loss_high",
                    False
                ):
                    score *= 4.0

                if evidence.get(
                    "response_time_high",
                    False
                ):
                    score *= 1.5

            # ==============================================
            # DATABASE FAILURE
            # ==============================================

            elif fault == FaultType.DATABASE_FAILURE:

                if evidence.get(
                    "db_latency_high",
                    False
                ):
                    score *= 4.0

                if evidence.get(
                    "db_failure_high",
                    False
                ):
                    score *= 4.0

                if evidence.get(
                    "response_time_high",
                    False
                ):
                    score *= 1.5

            # ==============================================
            # SERVICE FAILURE
            # ==============================================

            elif fault == FaultType.SERVICE_FAILURE:

                if evidence.get(
                    "availability_low",
                    False
                ):
                    score *= 4.0

                if evidence.get(
                    "error_rate_high",
                    False
                ):
                    score *= 2.0

            # ==============================================
            # HEALTH PROBE EVIDENCE
            # ==============================================

            application = evidence.get(
                "application"
            )

            network = evidence.get(
                "network"
            )

            database = evidence.get(
                "database"
            )

            if (
                fault in [
                    FaultType.CPU_OVERLOAD,
                    FaultType.MEMORY_LEAK,
                    FaultType.SERVICE_FAILURE
                ]
                and application in [
                    "DEGRADED",
                    "FAILED"
                ]
            ):
                score *= 2.0

            if (
                fault == FaultType.NETWORK_DEGRADATION
                and network == "FAILING"
            ):
                score *= 4.0

            if (
                fault == FaultType.DATABASE_FAILURE
                and database == "FAILING"
            ):
                score *= 4.0

            # ==============================================
            # DEEP DIAGNOSTIC EVIDENCE
            # ==============================================

            identified_fault = evidence.get(
                "identified_fault"
            )

            if identified_fault == fault.name:

                score *= 10.0

            evidence_scores[
                fault.value
            ] = score

        # --------------------------------------------------
        # RELIABILITY / CONFIDENCE
        # --------------------------------------------------

        if reliable:

            strength = np.clip(
                confidence,
                0.0,
                1.0
            )

        else:

            strength = 0.20

        # --------------------------------------------------
        # POSTERIOR
        # --------------------------------------------------

        posterior = (
            self.belief
            * evidence_scores
        )

        posterior_sum = np.sum(
            posterior
        )

        if posterior_sum > 0:

            posterior /= posterior_sum

        else:

            posterior = (
                np.ones(5) / 5.0
            )

        # --------------------------------------------------
        # BELIEF UPDATE
        # --------------------------------------------------

        updated_belief = (
            (1.0 - strength)
            * self.belief
            + strength
            * posterior
        )

        updated_belief /= np.sum(
            updated_belief
        )

        self.belief = (
            updated_belief.astype(
                np.float32
            )
        )

    # ======================================================
    # UNCERTAINTY
    # ======================================================

    def _calculate_uncertainty(self):

        entropy = -np.sum(
            self.belief
            * np.log(
                self.belief
                + 1e-10
            )
        )

        normalized_entropy = (
            entropy
            / np.log(5)
        )

        return float(
            np.clip(
                normalized_entropy,
                0.0,
                1.0
            )
        )

    # ======================================================
    # OBSERVATION
    # ======================================================

    def _get_observation(self):

        state = self.system.get_state()

        # --------------------------------------------------
        # SYSTEM METRICS
        # --------------------------------------------------

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

            state.db_failure_rate / 100.0

        ], dtype=np.float32)

        # --------------------------------------------------
        # UNCERTAINTY
        # --------------------------------------------------

        uncertainty = (
            self._calculate_uncertainty()
        )

        # --------------------------------------------------
        # EXTRA STATE VARIABLES
        # --------------------------------------------------

        extra_values = np.array([

            uncertainty,

            self.remaining_budget
            / self.initial_budget,

            min(
                self.accumulated_cost
                / 20.0,
                1.0
            ),

            min(
                self.query_count
                / 10.0,
                1.0
            ),

            min(
                self.elapsed_time
                / 60.0,
                1.0
            ),

            float(self.pending_verification),

            min(self.healthy_verification_streak / 2.0, 1.0)

        ], dtype=np.float32)

        # --------------------------------------------------
        # FINAL OBSERVATION
        # --------------------------------------------------

        observation = np.concatenate([

            system_values,

            self.belief,

            extra_values

        ])

        observation = np.clip(
            observation,
            0.0,
            1.0
        )

        return observation.astype(
            np.float32
        )


# ==========================================================
# BASIC TEST
# ==========================================================

if __name__ == "__main__":

    env = FaultDiagnosisEnv(
        seed=42
    )

    observation, info = env.reset()

    print("=" * 70)
    print("ENVIRONMENT TEST")
    print("=" * 70)

    print(
        "Hidden fault:",
        info["hidden_fault"]
    )

    print(
        "Severity:",
        info["severity"]
    )

    print(
        "\nObservation shape:",
        observation.shape
    )

    print(
        "\nInitial belief:",
        observation[11:16]
    )

    print(
        "\nInitial uncertainty:",
        observation[16]
    )

    print(
        "\nInitial budget:",
        env.remaining_budget
    )

    # ======================================================
    # TELEMETRY
    # ======================================================

    print("\n" + "=" * 70)
    print("ACTION: TELEMETRY")
    print("=" * 70)

    (
        observation,
        reward,
        terminated,
        truncated,
        info
    ) = env.step(0)

    print(
        "Reward:",
        reward
    )

    print(
        "Diagnostic result:"
    )

    print(
        info.get(
            "diagnostic_result"
        )
    )

    print(
        "Updated belief:",
        observation[11:16]
    )

    print(
        "Updated uncertainty:",
        observation[16]
    )

    print(
        "Remaining budget:",
        env.remaining_budget
    )

    print(
        "Terminated:",
        terminated
    )

    print(
        "Truncated:",
        truncated
    )

    # ======================================================
    # HEALTH PROBE
    # ======================================================

    print("\n" + "=" * 70)
    print("ACTION: HEALTH PROBE")
    print("=" * 70)

    (
        observation,
        reward,
        terminated,
        truncated,
        info
    ) = env.step(1)

    print(
        "Reward:",
        reward
    )

    print(
        "Diagnostic result:"
    )

    print(
        info.get(
            "diagnostic_result"
        )
    )

    print(
        "Updated belief:",
        observation[11:16]
    )

    print(
        "Remaining budget:",
        env.remaining_budget
    )

    print(
        "Terminated:",
        terminated
    )

    print(
        "Truncated:",
        truncated
    )

    # ======================================================
    # DEEP DIAGNOSTIC
    # ======================================================

    print("\n" + "=" * 70)
    print("ACTION: DEEP DIAGNOSTIC")
    print("=" * 70)

    (
        observation,
        reward,
        terminated,
        truncated,
        info
    ) = env.step(3)

    print(
        "Reward:",
        reward
    )

    print(
        "Diagnostic result:"
    )

    print(
        info.get(
            "diagnostic_result"
        )
    )

    print(
        "Updated belief:",
        observation[11:16]
    )

    print(
        "Remaining budget:",
        env.remaining_budget
    )

    print(
        "Terminated:",
        terminated
    )

    print(
        "Truncated:",
        truncated
    )

    env.close()