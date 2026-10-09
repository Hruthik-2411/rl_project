from enum import Enum
import numpy as np


class FaultType(Enum):
    CPU_OVERLOAD = 0
    MEMORY_LEAK = 1
    NETWORK_DEGRADATION = 2
    DATABASE_FAILURE = 3
    SERVICE_FAILURE = 4


class FaultDynamics:

    def __init__(self, system, seed=None):
        self.system = system
        self.rng = np.random.default_rng(seed)

        self.fault = None
        self.severity = 0.0

        # How quickly the fault develops
        self.progression_rate = 0.0

    def inject_fault(self, fault, severity=1.0):
        """
        Start a fault with an initial severity.

        severity:
            0.0 = healthy
            1.0 = severe fault
        """

        self.fault = fault
        self.severity = float(np.clip(severity, 0.1, 1.0))

        # Different faults develop at different speeds
        progression_rates = {
            FaultType.CPU_OVERLOAD: 0.08,
            FaultType.MEMORY_LEAK: 0.04,
            FaultType.NETWORK_DEGRADATION: 0.06,
            FaultType.DATABASE_FAILURE: 0.07,
            FaultType.SERVICE_FAILURE: 0.10
        }

        self.progression_rate = progression_rates[fault]

    def clear_fault(self):
        """Remove the active fault."""

        self.fault = None
        self.severity = 0.0
        self.progression_rate = 0.0

    def apply_fault(self):
        """
        Progress the fault and apply its effects
        to the distributed system.
        """

        if self.fault is None:
            return

        # Fault becomes gradually worse
        self.severity += self.progression_rate

        # Small stochastic variation
        self.severity += self.rng.normal(0, 0.005)

        self.severity = float(
            np.clip(self.severity, 0.0, 1.0)
        )

        if self.fault == FaultType.CPU_OVERLOAD:
            self._cpu_overload()

        elif self.fault == FaultType.MEMORY_LEAK:
            self._memory_leak()

        elif self.fault == FaultType.NETWORK_DEGRADATION:
            self._network_degradation()

        elif self.fault == FaultType.DATABASE_FAILURE:
            self._database_failure()

        elif self.fault == FaultType.SERVICE_FAILURE:
            self._service_failure()

    def _cpu_overload(self):

        severity = self.severity

        # Excessive workload saturates CPU
        self.system.state.cpu += 5.0 * severity

        # CPU saturation causes processing delay
        self.system.state.response_time += 2.5 * severity

        # Requests start waiting
        self.system.state.queue_length += 3.0 * severity

        # High CPU eventually causes request failures
        self.system.state.error_rate += 0.8 * severity

    def _memory_leak(self):

        severity = self.severity

        # Main signature of memory leak
        self.system.state.memory += 3.5 * severity

        # Memory pressure increases CPU usage
        self.system.state.cpu += 0.8 * severity

        # Application becomes slower
        self.system.state.response_time += 1.8 * severity

        self.system.state.queue_length += 1.5 * severity

        self.system.state.error_rate += 0.4 * severity

    def _network_degradation(self):

        severity = self.severity

        # Network itself is degraded
        self.system.state.network_latency += 7.0 * severity

        self.system.state.packet_loss += 2.0 * severity

        # Network problems propagate to application
        self.system.state.response_time += 3.5 * severity

        self.system.state.error_rate += 1.2 * severity

        self.system.state.queue_length += 2.0 * severity

    def _database_failure(self):

        severity = self.severity

        # Database becomes slower/unavailable
        self.system.state.db_latency += 9.0 * severity

        self.system.state.db_failure_rate += 6.0 * severity

        # Application waits for database responses
        self.system.state.response_time += 5.0 * severity

        self.system.state.queue_length += 3.0 * severity

        self.system.state.error_rate += 2.0 * severity

        # Some CPU overhead from waiting/retries
        self.system.state.cpu += 1.0 * severity

    def _service_failure(self):

        severity = self.severity

        # Service becomes progressively unavailable
        self.system.state.availability -= 5.0 * severity

        # Requests fail
        self.system.state.error_rate += 5.0 * severity

        # Failed/blocked requests accumulate
        self.system.state.queue_length += 3.0 * severity

        # Failed service causes long response times
        self.system.state.response_time += 6.0 * severity

    def is_critical(self):
        """
        Determine whether the system has reached
        a critical condition.
        """

        state = self.system.state

        return (
            state.availability < 30
            or state.error_rate > 70
            or state.cpu > 95
            or state.memory > 95
            or state.network_latency > 300
            or state.db_failure_rate > 70
        )


if __name__ == "__main__":

    from system_model import DistributedSystem

    faults = list(FaultType)

    for fault in faults:

        print("\n" + "=" * 60)
        print(f"FAULT: {fault.name}")
        print("=" * 60)

        system = DistributedSystem(seed=42)
        system.reset()

        fault_system = FaultDynamics(
            system,
            seed=42
        )

        fault_system.inject_fault(
            fault,
            severity=0.3
        )

        for t in range(10):

            system.step_time()
            fault_system.apply_fault()

            state = system.get_state()

            print(
                f"t={t + 1:02d} | "
                f"severity={fault_system.severity:.2f} | "
                f"CPU={state.cpu:.1f} | "
                f"MEM={state.memory:.1f} | "
                f"NET={state.network_latency:.1f} | "
                f"DB={state.db_latency:.1f} | "
                f"ERR={state.error_rate:.1f} | "
                f"AVAIL={state.availability:.1f}"
            )