from enum import Enum
import numpy as np

from environment.faults import FaultType


class RecoveryAction(Enum):
    RESTART_SERVICE = 0
    REROUTE_TRAFFIC = 1
    ISOLATE_COMPONENT = 2
    REDUCE_LOAD = 3
    ROLLBACK = 4
    STOP_SYSTEM = 5


class RecoveryManager:

    def __init__(self, system, fault_system, seed=None):

        self.system = system
        self.fault_system = fault_system

        self.rng = np.random.default_rng(seed)

        self.actions = {

            RecoveryAction.RESTART_SERVICE: {
                "cost": 3.0,
                "delay": 2,
                "success_probability": 0.90,
                "effective_faults": [
                    FaultType.SERVICE_FAILURE,
                    FaultType.MEMORY_LEAK,
                    FaultType.DATABASE_FAILURE
                ]
            },

            RecoveryAction.REROUTE_TRAFFIC: {
                "cost": 2.0,
                "delay": 1,
                "success_probability": 0.90,
                "effective_faults": [
                    FaultType.NETWORK_DEGRADATION
                ]
            },

            RecoveryAction.ISOLATE_COMPONENT: {
                "cost": 3.0,
                "delay": 2,
                "success_probability": 0.85,
                "effective_faults": [
                    FaultType.DATABASE_FAILURE,
                    FaultType.SERVICE_FAILURE
                ]
            },

            RecoveryAction.REDUCE_LOAD: {
                "cost": 2.0,
                "delay": 1,
                "success_probability": 0.90,
                "effective_faults": [
                    FaultType.CPU_OVERLOAD
                ]
            },

            RecoveryAction.ROLLBACK: {
                "cost": 4.0,
                "delay": 3,
                "success_probability": 0.85,
                "effective_faults": [
                    FaultType.MEMORY_LEAK,
                    FaultType.SERVICE_FAILURE
                ]
            },

            RecoveryAction.STOP_SYSTEM: {
                "cost": 1.0,
                "delay": 1,
                "success_probability": 1.00,
                "effective_faults": list(FaultType)
            }
        }

    # =====================================================
    # EXECUTE RECOVERY
    # =====================================================

    def execute(self, action):

        if action not in self.actions:
            raise ValueError(f"Invalid recovery action: {action}")

        config = self.actions[action]

        hidden_fault = self.fault_system.fault

        severity_before = self.fault_system.severity

        # -------------------------------------------------
        # STOP SYSTEM
        # -------------------------------------------------

        if action == RecoveryAction.STOP_SYSTEM:

            self._apply_stop()

            return {
                "action": action.name,
                "success": True,
                "safe_termination": True,
                "cost": config["cost"],
                "delay": config["delay"],
                "severity_before": severity_before,
                "severity_after": 0.0,
                "message": "System safely stopped."
            }

        # -------------------------------------------------
        # Check whether action is appropriate
        # -------------------------------------------------

        correct_fault = (
            hidden_fault in config["effective_faults"]
        )

        # -------------------------------------------------
        # Wrong recovery action
        # -------------------------------------------------

        if not correct_fault:

            self.fault_system.severity = min(
                1.0,
                self.fault_system.severity + 0.05
            )

            return {
                "action": action.name,
                "success": False,
                "safe_termination": False,
                "cost": config["cost"],
                "delay": config["delay"],
                "severity_before": severity_before,
                "severity_after": self.fault_system.severity,
                "message": (
                    f"{action.name} was not appropriate "
                    f"for the current fault."
                )
            }

        # -------------------------------------------------
        # Correct recovery
        # -------------------------------------------------

        success = (
            self.rng.random()
            < config["success_probability"]
        )

        if success:

            # Reduce fault severity substantially.
            self.fault_system.severity *= 0.10

            self._apply_recovery_effect(action)

            recovered = (
                self.fault_system.severity < 0.20
            )

            # Keep the fault model active at low severity until the
            # environment verifies service health. This allows a failed
            # health check to return to diagnosis/recovery instead of
            # leaving the environment with no active fault to diagnose.

            return {
                "action": action.name,
                "success": recovered,
                "safe_termination": False,
                "cost": config["cost"],
                "delay": config["delay"],
                "severity_before": severity_before,
                "severity_after": self.fault_system.severity,
                "message": (
                    f"{action.name} successfully reduced "
                    f"the fault severity."
                )
            }

        # -------------------------------------------------
        # Correct action but recovery failed
        # -------------------------------------------------

        self.fault_system.severity = min(
            1.0,
            self.fault_system.severity + 0.03
        )

        return {
            "action": action.name,
            "success": False,
            "safe_termination": False,
            "cost": config["cost"],
            "delay": config["delay"],
            "severity_before": severity_before,
            "severity_after": self.fault_system.severity,
            "message": (
                f"{action.name} was appropriate, "
                f"but the recovery attempt failed."
            )
        }

    # =====================================================
    # RECOVERY EFFECTS
    # =====================================================

    def _apply_recovery_effect(self, action):

        state = self.system.state

        if action == RecoveryAction.RESTART_SERVICE:

            state.error_rate *= 0.40
            state.response_time *= 0.70
            state.queue_length *= 0.60

            state.availability = min(
                100.0,
                state.availability + 10.0
            )

        elif action == RecoveryAction.REROUTE_TRAFFIC:

            state.network_latency *= 0.50
            state.packet_loss *= 0.40
            state.response_time *= 0.70
            state.error_rate *= 0.60

        elif action == RecoveryAction.ISOLATE_COMPONENT:

            state.db_failure_rate *= 0.30
            state.db_latency *= 0.60
            state.error_rate *= 0.60
            state.queue_length *= 0.70

        elif action == RecoveryAction.REDUCE_LOAD:

            state.request_rate *= 0.60

            # Correct field name from SystemState
            state.cpu *= 0.65

            state.queue_length *= 0.50
            state.response_time *= 0.70
            state.error_rate *= 0.60

        elif action == RecoveryAction.ROLLBACK:

            state.error_rate *= 0.50
            state.response_time *= 0.70
            state.queue_length *= 0.60

            state.availability = min(
                100.0,
                state.availability + 8.0
            )

        self._clip_state()

    # =====================================================
    # STOP SYSTEM
    # =====================================================

    def _apply_stop(self):

        state = self.system.state

        state.request_rate = 0.0
        state.queue_length = 0.0
        state.cpu = 0.0
        state.response_time = 0.0
        state.error_rate = 0.0
        state.availability = 0.0

        self.fault_system.clear_fault()

    # =====================================================
    # KEEP VALUES VALID
    # =====================================================

    def _clip_state(self):

        state = self.system.state

        state.cpu = float(np.clip(state.cpu, 0, 100))
        state.memory = float(np.clip(state.memory, 0, 100))
        state.request_rate = float(
            np.clip(state.request_rate, 0, 200)
        )
        state.response_time = float(
            np.clip(state.response_time, 1, 500)
        )
        state.error_rate = float(
            np.clip(state.error_rate, 0, 100)
        )
        state.queue_length = float(
            np.clip(state.queue_length, 0, 200)
        )
        state.availability = float(
            np.clip(state.availability, 0, 100)
        )
        state.network_latency = float(
            np.clip(state.network_latency, 1, 500)
        )
        state.packet_loss = float(
            np.clip(state.packet_loss, 0, 100)
        )
        state.db_latency = float(
            np.clip(state.db_latency, 1, 500)
        )
        state.db_failure_rate = float(
            np.clip(state.db_failure_rate, 0, 100)
        )


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    from environment.system_model import DistributedSystem
    from environment.faults import FaultDynamics

    print("=" * 60)
    print("RECOVERY TEST")
    print("=" * 60)

    # Create system
    system = DistributedSystem(seed=42)

    # IMPORTANT: initialize SystemState
    system.reset()

    # Create fault system
    fault_system = FaultDynamics(
        system,
        seed=42
    )

    # Inject CPU overload
    fault_system.inject_fault(
        FaultType.CPU_OVERLOAD,
        severity=0.8
    )

    # Create recovery manager
    recovery = RecoveryManager(
        system,
        fault_system,
        seed=42
    )

    print("\nHidden fault:")
    print(fault_system.fault.name)

    print("\nInitial severity:")
    print(fault_system.severity)

    print("\nInitial CPU:")
    print(system.state.cpu)

    print("\nExecuting REDUCE_LOAD...")

    result = recovery.execute(
        RecoveryAction.REDUCE_LOAD
    )

    print("\nRecovery result:")
    print(result)

    print("\nFinal severity:")
    print(fault_system.severity)

    print("\nFinal CPU:")
    print(system.state.cpu)

    print("\nFault after recovery:")
    print(
        fault_system.fault.name
        if fault_system.fault is not None
        else "NO ACTIVE FAULT"
    )