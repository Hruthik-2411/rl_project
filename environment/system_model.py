from dataclasses import dataclass
import numpy as np


@dataclass
class SystemState:
    # Application
    cpu: float
    memory: float
    request_rate: float
    response_time: float
    error_rate: float
    queue_length: float
    availability: float

    # Network
    network_latency: float
    packet_loss: float

    # Database
    db_latency: float
    db_failure_rate: float


class DistributedSystem:
    """
    Simulates a distributed IT system consisting of:

        Client traffic
              |
        Application
          /       \
      Network    Database

    The system has normal operating dynamics and small
    environmental noise. Faults are applied separately
    by FaultDynamics.
    """

    def __init__(self, seed=None):

        self.rng = np.random.default_rng(seed)

        self.state = None
        self.time = 0

    def reset(self):

        self.time = 0

        self.state = SystemState(

            # Application
            cpu=self.rng.uniform(40, 55),
            memory=self.rng.uniform(45, 60),
            request_rate=self.rng.uniform(80, 120),
            response_time=self.rng.uniform(50, 80),
            error_rate=self.rng.uniform(0, 1),
            queue_length=self.rng.uniform(10, 30),
            availability=self.rng.uniform(99, 100),

            # Network
            network_latency=self.rng.uniform(50, 80),
            packet_loss=self.rng.uniform(0, 1),

            # Database
            db_latency=self.rng.uniform(30, 60),
            db_failure_rate=self.rng.uniform(0, 1)
        )

        return self.state

    def get_state(self):
        return self.state

    def step_time(self):

        if self.state is None:
            self.reset()

        self.time += 1

        # --------------------------------------------------
        # 1. Request-rate variation
        # --------------------------------------------------

        self.state.request_rate += self.rng.normal(0, 3)

        # Keep realistic workload
        self.state.request_rate = np.clip(
            self.state.request_rate,
            50,
            160
        )

        # --------------------------------------------------
        # 2. CPU dynamics
        # --------------------------------------------------

        # Higher traffic naturally increases CPU utilization
        target_cpu = (
            35
            + 0.20 * self.state.request_rate
        )

        self.state.cpu += (
            0.05 * (target_cpu - self.state.cpu)
            + self.rng.normal(0, 0.5)
        )

        # --------------------------------------------------
        # 3. Memory dynamics
        # --------------------------------------------------

        # Memory changes slowly under normal operation
        self.state.memory += (
            0.01 * (55 - self.state.memory)
            + self.rng.normal(0, 0.15)
        )

        # --------------------------------------------------
        # 4. Network dynamics
        # --------------------------------------------------

        self.state.network_latency += (
            0.05 * (65 - self.state.network_latency)
            + self.rng.normal(0, 1.0)
        )

        self.state.packet_loss += (
            0.05 * (0.5 - self.state.packet_loss)
            + self.rng.normal(0, 0.03)
        )

        # --------------------------------------------------
        # 5. Database dynamics
        # --------------------------------------------------

        self.state.db_latency += (
            0.05 * (45 - self.state.db_latency)
            + self.rng.normal(0, 0.7)
        )

        self.state.db_failure_rate += (
            0.05 * (0.5 - self.state.db_failure_rate)
            + self.rng.normal(0, 0.03)
        )

        # --------------------------------------------------
        # 6. Application response time
        # --------------------------------------------------

        # Response time depends on:
        # CPU pressure
        # Network latency
        # Database latency

        cpu_effect = max(
            0,
            self.state.cpu - 50
        ) * 0.8

        network_effect = max(
            0,
            self.state.network_latency - 60
        ) * 0.35

        database_effect = max(
            0,
            self.state.db_latency - 45
        ) * 0.45

        target_response_time = (
            55
            + cpu_effect
            + network_effect
            + database_effect
        )

        self.state.response_time += (
            0.15 *
            (target_response_time - self.state.response_time)
            + self.rng.normal(0, 1.5)
        )

        # --------------------------------------------------
        # 7. Queue dynamics
        # --------------------------------------------------

        # Queue grows when response time becomes high
        queue_pressure = max(
            0,
            self.state.response_time - 80
        ) * 0.15

        self.state.queue_length += (
            queue_pressure
            - 0.10 * self.state.queue_length
            + self.rng.normal(0, 0.5)
        )

        # --------------------------------------------------
        # 8. Error-rate dynamics
        # --------------------------------------------------

        error_pressure = (

            max(0, self.state.response_time - 100)
            * 0.04

            + max(0, self.state.packet_loss - 1)
            * 0.5

            + max(0, self.state.db_failure_rate - 2)
            * 0.3
        )

        self.state.error_rate += (
            error_pressure
            - 0.08 * self.state.error_rate
            + self.rng.normal(0, 0.05)
        )

        # --------------------------------------------------
        # 9. Availability dynamics
        # --------------------------------------------------

        availability_loss = (
            self.state.error_rate * 0.01
            + max(
                0,
                self.state.db_failure_rate - 5
            ) * 0.02
        )

        target_availability = (
            100 - availability_loss
        )

        self.state.availability += (
            0.10 *
            (target_availability - self.state.availability)
            + self.rng.normal(0, 0.03)
        )

        # --------------------------------------------------
        # Keep all values physically meaningful
        # --------------------------------------------------

        self._clip_values()

        return self.state

    def _clip_values(self):

        self.state.cpu = np.clip(
            self.state.cpu,
            0,
            100
        )

        self.state.memory = np.clip(
            self.state.memory,
            0,
            100
        )

        self.state.request_rate = np.clip(
            self.state.request_rate,
            0,
            200
        )

        self.state.response_time = np.clip(
            self.state.response_time,
            1,
            500
        )

        self.state.error_rate = np.clip(
            self.state.error_rate,
            0,
            100
        )

        self.state.queue_length = np.clip(
            self.state.queue_length,
            0,
            200
        )

        self.state.availability = np.clip(
            self.state.availability,
            0,
            100
        )

        self.state.network_latency = np.clip(
            self.state.network_latency,
            1,
            500
        )

        self.state.packet_loss = np.clip(
            self.state.packet_loss,
            0,
            100
        )

        self.state.db_latency = np.clip(
            self.state.db_latency,
            1,
            500
        )

        self.state.db_failure_rate = np.clip(
            self.state.db_failure_rate,
            0,
            100
        )


if __name__ == "__main__":

    system = DistributedSystem(seed=42)

    system.reset()

    print("=" * 70)
    print("NORMAL DISTRIBUTED SYSTEM")
    print("=" * 70)

    for t in range(10):

        state = system.step_time()

        print(
            f"t={t + 1:02d} | "
            f"CPU={state.cpu:5.1f}% | "
            f"MEM={state.memory:5.1f}% | "
            f"REQ={state.request_rate:5.1f} | "
            f"RESP={state.response_time:6.1f}ms | "
            f"ERR={state.error_rate:5.2f}% | "
            f"QUEUE={state.queue_length:5.1f} | "
            f"NET={state.network_latency:6.1f}ms | "
            f"LOSS={state.packet_loss:4.2f}% | "
            f"DB={state.db_latency:6.1f}ms | "
            f"DBFAIL={state.db_failure_rate:4.2f}% | "
            f"AVAIL={state.availability:5.2f}%"
        )