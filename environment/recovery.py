
from enum import IntEnum

import numpy as np


class RecoveryAction(IntEnum):
    """Recovery actions used by the RL environment."""

    RESTART_SERVICE = 0
    REROUTE_TRAFFIC = 1
    ISOLATE_COMPONENT = 2
    REDUCE_LOAD = 3
    ROLLBACK = 4
    STOP_SYSTEM = 5


class RecoveryManager:
    """
    Executes recovery interventions against the simulated distributed system.

    An intervention being applied successfully does not guarantee that the
    system is healthy. The RL environment must verify recovery separately.
    """

    ACTION_CONFIG = {
        RecoveryAction.RESTART_SERVICE: {
            "cost": 3.0,
            "delay": 2,
            "success_probability": 0.88,
            "effective_faults": {"SERVICE_FAILURE", "MEMORY_LEAK", "DATABASE_FAILURE"},
        },
        RecoveryAction.REROUTE_TRAFFIC: {
            "cost": 2.0,
            "delay": 1,
            "success_probability": 0.90,
            "effective_faults": {"NETWORK_DEGRADATION"},
        },
        RecoveryAction.ISOLATE_COMPONENT: {
            "cost": 3.0,
            "delay": 2,
            "success_probability": 0.82,
            "effective_faults": {"DATABASE_FAILURE", "SERVICE_FAILURE"},
        },
        RecoveryAction.REDUCE_LOAD: {
            "cost": 2.0,
            "delay": 1,
            "success_probability": 0.88,
            "effective_faults": {"CPU_OVERLOAD"},
        },
        RecoveryAction.ROLLBACK: {
            "cost": 4.0,
            "delay": 3,
            "success_probability": 0.82,
            "effective_faults": {"MEMORY_LEAK", "SERVICE_FAILURE"},
        },
        RecoveryAction.STOP_SYSTEM: {
            "cost": 1.0,
            "delay": 1,
            "success_probability": 1.0,
            "effective_faults": {
                "CPU_OVERLOAD",
                "MEMORY_LEAK",
                "NETWORK_DEGRADATION",
                "DATABASE_FAILURE",
                "SERVICE_FAILURE",
            },
        },
    }

    def __init__(self, system, fault_system, seed=None):
        self.system = system
        self.fault_system = fault_system
        self.rng = np.random.default_rng(seed)

    def execute(self, action):
        """Execute one recovery action and return structured result metadata."""
        try:
            action = RecoveryAction(action)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid recovery action: {action!r}")

        config = self.ACTION_CONFIG[action]
        state = self.system.get_state()

        severity_before = float(
            getattr(self.fault_system, "severity", 0.0)
        )

        if action == RecoveryAction.STOP_SYSTEM:
            self._apply_stop()
            return {
                "action": action.name,
                "success": True,
                "safe_termination": True,
                "intervention_applied": True,
                "cost": config["cost"],
                "delay": config["delay"],
                "severity_before": severity_before,
                "severity_after": 0.0,
                "message": "System safely stopped to contain the fault.",
            }

        active_fault = getattr(self.fault_system, "fault", None)
        active_fault_name = getattr(active_fault, "name", str(active_fault))

        # A recovery action is appropriate only when it matches the active
        # fault. The active fault is used by the simulator, not the agent.
        correct_action = active_fault_name in config["effective_faults"]

        if correct_action:
            success = bool(
                self.rng.random() < config["success_probability"]
            )
        else:
            success = False

        if success:
            severity_after = max(0.10, severity_before * 0.40)
            self.fault_system.severity = severity_after
            self._apply_action_effects(action, state)

            message = (
                f"{action.name} was applied successfully; "
                "system health still requires verification."
            )
        else:
            # A wrong or unsuccessful intervention can worsen the fault.
            severity_after = min(1.0, severity_before + 0.08)
            if active_fault is not None:
                self.fault_system.severity = severity_after

            message = (
                f"{action.name} did not resolve the fault. "
                "Further diagnosis or a different intervention is needed."
            )

        self._clip_state(state)

        return {
            "action": action.name,
            "success": success,
            "safe_termination": False,
            "intervention_applied": success,
            "cost": config["cost"],
            "delay": config["delay"],
            "severity_before": severity_before,
            "severity_after": severity_after,
            "message": message,
        }

    def _apply_action_effects(self, action, state):
        """Apply partial improvements to observable system metrics."""

        if action == RecoveryAction.RESTART_SERVICE:
            state.error_rate *= 0.65
            state.response_time *= 0.80
            state.queue_length *= 0.75
            state.availability = min(100.0, state.availability + 4.0)
            state.memory *= 0.85

        elif action == RecoveryAction.REROUTE_TRAFFIC:
            state.network_latency *= 0.65
            state.packet_loss *= 0.60
            state.response_time *= 0.82
            state.error_rate *= 0.80

        elif action == RecoveryAction.ISOLATE_COMPONENT:
            state.db_failure_rate *= 0.55
            state.db_latency *= 0.75
            state.error_rate *= 0.80
            state.queue_length *= 0.85

        elif action == RecoveryAction.REDUCE_LOAD:
            state.request_rate *= 0.80
            state.cpu *= 0.80
            state.queue_length *= 0.75
            state.response_time *= 0.82
            state.error_rate *= 0.80

        elif action == RecoveryAction.ROLLBACK:
            state.error_rate *= 0.70
            state.response_time *= 0.82
            state.queue_length *= 0.80
            state.memory *= 0.85
            state.availability = min(100.0, state.availability + 3.0)

    def _apply_stop(self):
        """Contain the fault by stopping the simulated service."""
        state = self.system.get_state()

        state.request_rate = 0.0
        state.queue_length = 0.0
        state.cpu = 0.0
        state.response_time = 0.0
        state.error_rate = 0.0
        state.availability = 0.0

        if hasattr(self.fault_system, "clear_fault"):
            self.fault_system.clear_fault()

        self._clip_state(state)

    @staticmethod
    def _clip_state(state):
        """Keep system metrics within their expected ranges."""
        bounds = {
            "cpu": (0.0, 100.0),
            "memory": (0.0, 100.0),
            "request_rate": (0.0, 200.0),
            "response_time": (0.0, 500.0),
            "error_rate": (0.0, 100.0),
            "queue_length": (0.0, 200.0),
            "availability": (0.0, 100.0),
            "network_latency": (0.0, 500.0),
            "packet_loss": (0.0, 100.0),
            "db_latency": (0.0, 500.0),
            "db_failure_rate": (0.0, 100.0),
        }

        for name, (lower, upper) in bounds.items():
            if hasattr(state, name):
                value = float(getattr(state, name))
                setattr(state, name, float(np.clip(value, lower, upper)))
