from enum import Enum
import numpy as np


class DiagnosticSource(Enum):
    TELEMETRY = 0
    HEALTH_PROBE = 1
    LOGS = 2
    DEEP_DIAGNOSTIC = 3


class DiagnosticManager:

    def __init__(self, seed=None):

        self.rng = np.random.default_rng(seed)

        self.sources = {

            DiagnosticSource.TELEMETRY: {
                "cost": 1,
                "delay": 0,
                "reliability": 0.80,
                "availability": 0.99
            },

            DiagnosticSource.HEALTH_PROBE: {
                "cost": 2,
                "delay": 1,
                "reliability": 0.90,
                "availability": 0.95
            },

            DiagnosticSource.LOGS: {
                "cost": 3,
                "delay": 2,
                "reliability": 0.85,
                "availability": 0.90
            },

            DiagnosticSource.DEEP_DIAGNOSTIC: {
                "cost": 6,
                "delay": 3,
                "reliability": 0.98,
                "availability": 0.85
            }
        }

    def query(self, source, fault):

        config = self.sources[source]

        # --------------------------------------------------
        # Check whether diagnostic source is available
        # --------------------------------------------------

        available = (
            self.rng.random()
            < config["availability"]
        )

        if not available:

            return {
                "source": source.name,
                "available": False,
                "evidence": {},
                "confidence": 0.0,
                "cost": config["cost"],
                "delay": config["delay"],
                "reliable": False
            }

        # --------------------------------------------------
        # Reliability
        # --------------------------------------------------

        reliable = (
            self.rng.random()
            < config["reliability"]
        )

        # --------------------------------------------------
        # Generate evidence
        # --------------------------------------------------

        evidence = self._generate_evidence(
            source,
            fault,
            reliable
        )

        confidence = self._calculate_confidence(
            source,
            fault,
            reliable
        )

        return {
            "source": source.name,
            "available": True,
            "evidence": evidence,
            "confidence": confidence,
            "cost": config["cost"],
            "delay": config["delay"],
            "reliable": reliable
        }

    # ======================================================
    # Evidence generation
    # ======================================================

    def _generate_evidence(
        self,
        source,
        fault,
        reliable
    ):

        fault_name = fault.name

        # --------------------------------------------------
        # TELEMETRY
        # --------------------------------------------------

        if source == DiagnosticSource.TELEMETRY:

            evidence = self._telemetry_evidence(
                fault_name
            )

        # --------------------------------------------------
        # HEALTH PROBE
        # --------------------------------------------------

        elif source == DiagnosticSource.HEALTH_PROBE:

            evidence = self._health_probe_evidence(
                fault_name
            )

        # --------------------------------------------------
        # LOGS
        # --------------------------------------------------

        elif source == DiagnosticSource.LOGS:

            evidence = self._log_evidence(
                fault_name
            )

        # --------------------------------------------------
        # DEEP DIAGNOSTIC
        # --------------------------------------------------

        else:

            evidence = self._deep_diagnostic_evidence(
                fault_name
            )

        # --------------------------------------------------
        # If unreliable, corrupt some evidence
        # --------------------------------------------------

        if not reliable:

            evidence = self._corrupt_evidence(
                evidence
            )

        return evidence

    # ======================================================
    # Telemetry
    # ======================================================

    def _telemetry_evidence(self, fault):

        evidence = {

            "cpu_high": False,
            "memory_high": False,
            "network_latency_high": False,
            "packet_loss_high": False,
            "db_latency_high": False,
            "db_failure_high": False,
            "response_time_high": False,
            "error_rate_high": False,
            "availability_low": False
        }

        if fault == "CPU_OVERLOAD":

            evidence["cpu_high"] = True
            evidence["response_time_high"] = True
            evidence["error_rate_high"] = True

        elif fault == "MEMORY_LEAK":

            evidence["memory_high"] = True
            evidence["response_time_high"] = True

        elif fault == "NETWORK_DEGRADATION":

            evidence["network_latency_high"] = True
            evidence["packet_loss_high"] = True
            evidence["response_time_high"] = True

        elif fault == "DATABASE_FAILURE":

            evidence["db_latency_high"] = True
            evidence["db_failure_high"] = True
            evidence["response_time_high"] = True

        elif fault == "SERVICE_FAILURE":

            evidence["availability_low"] = True
            evidence["error_rate_high"] = True

        return evidence

    # ======================================================
    # Health probe
    # ======================================================

    def _health_probe_evidence(self, fault):

        evidence = {

            "application": "HEALTHY",
            "network": "HEALTHY",
            "database": "HEALTHY"
        }

        if fault == "CPU_OVERLOAD":

            evidence["application"] = "DEGRADED"

        elif fault == "MEMORY_LEAK":

            evidence["application"] = "DEGRADED"

        elif fault == "NETWORK_DEGRADATION":

            evidence["network"] = "FAILING"

        elif fault == "DATABASE_FAILURE":

            evidence["database"] = "FAILING"

        elif fault == "SERVICE_FAILURE":

            evidence["application"] = "FAILED"

        return evidence

    # ======================================================
    # Logs
    # ======================================================

    def _log_evidence(self, fault):

        logs = {

            "CPU_OVERLOAD":
                [
                    "CPU saturation detected",
                    "Request queue increasing"
                ],

            "MEMORY_LEAK":
                [
                    "Memory consumption increasing",
                    "Memory pressure detected"
                ],

            "NETWORK_DEGRADATION":
                [
                    "Connection timeout detected",
                    "Packet retransmission detected"
                ],

            "DATABASE_FAILURE":
                [
                    "Database connection failure",
                    "Database request timeout"
                ],

            "SERVICE_FAILURE":
                [
                    "Service unavailable",
                    "Request processing failed"
                ]
        }

        return {
            "events": logs[fault]
        }

    # ======================================================
    # Deep diagnostic
    # ======================================================

    def _deep_diagnostic_evidence(self, fault):

        return {
            "identified_fault": fault,
            "diagnostic_confidence": 0.96
        }

    # ======================================================
    # Confidence
    # ======================================================

    def _calculate_confidence(
        self,
        source,
        fault,
        reliable
    ):

        if not reliable:
            return 0.20

        confidence_values = {

            DiagnosticSource.TELEMETRY: 0.60,

            DiagnosticSource.HEALTH_PROBE: 0.75,

            DiagnosticSource.LOGS: 0.85,

            DiagnosticSource.DEEP_DIAGNOSTIC: 0.96
        }

        return confidence_values[source]

    # ======================================================
    # Corrupt evidence
    # ======================================================

    def _corrupt_evidence(self, evidence):

        if not evidence:
            return evidence

        # Flip one boolean telemetry indicator
        boolean_keys = [
            key for key, value in evidence.items()
            if isinstance(value, bool)
        ]

        if boolean_keys:

            key = self.rng.choice(
                boolean_keys
            )

            evidence[key] = not evidence[key]

            return evidence

        # Corrupt health probe
        if "application" in evidence:

            evidence["application"] = "UNKNOWN"

        elif "network" in evidence:

            evidence["network"] = "UNKNOWN"

        elif "database" in evidence:

            evidence["database"] = "UNKNOWN"

        return evidence


if __name__ == "__main__":

    from system_model import DistributedSystem
    from faults import FaultDynamics, FaultType

    system = DistributedSystem(seed=42)

    system.reset()

    fault_system = FaultDynamics(
        system,
        seed=42
    )

    fault_system.inject_fault(
        FaultType.NETWORK_DEGRADATION,
        severity=0.6
    )

    diagnostics = DiagnosticManager(
        seed=42
    )

    print("=" * 60)
    print("HIDDEN FAULT")
    print("=" * 60)

    print(
        fault_system.fault.name
    )

    print("\n" + "=" * 60)
    print("DIAGNOSTIC RESULTS")
    print("=" * 60)

    for source in DiagnosticSource:

        result = diagnostics.query(
            source,
            fault_system.fault
        )

        print("\nSOURCE:", source.name)
        print(result)