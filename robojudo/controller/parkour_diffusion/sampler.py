"""NumPy Euler/Heun sampler for a noise-to-motion rectified flow."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


class FlowMatchingSampler:
    """Integrate the checkpoint velocity field from noise at t=1 to data at t=0."""

    def __init__(
        self,
        sampling_steps: int = 10,
        method: str = "euler",
        time_embedding_scale: float = 1000.0,
    ) -> None:
        if sampling_steps <= 0:
            raise ValueError("sampling_steps must be positive")
        if method not in ("euler", "heun"):
            raise ValueError(f"Flow sampler must be 'euler' or 'heun', got {method!r}")
        self.sampling_steps = int(sampling_steps)
        self.method = method
        self.time_embedding_scale = float(time_embedding_scale)

    def sample(
        self,
        velocity_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        initial_noise: np.ndarray,
    ) -> np.ndarray:
        x = np.asarray(initial_noise, dtype=np.float32)
        times = np.linspace(1.0, 0.0, self.sampling_steps + 1, dtype=np.float32)
        for index in range(self.sampling_steps):
            time = times[index]
            next_time = times[index + 1]
            dt = np.float32(next_time - time)
            model_time = np.asarray(
                [time * self.time_embedding_scale], dtype=np.float32
            )
            velocity = velocity_fn(x, model_time)
            if self.method == "euler":
                x = x + dt * velocity
                continue

            prediction = x + dt * velocity
            next_model_time = np.asarray(
                [next_time * self.time_embedding_scale], dtype=np.float32
            )
            next_velocity = velocity_fn(prediction, next_model_time)
            x = x + np.float32(0.5) * dt * (velocity + next_velocity)
        return np.asarray(x, dtype=np.float32)
