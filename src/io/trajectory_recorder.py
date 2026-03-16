from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass
class TrajectoryRecorderConfig:
    enabled: bool = True
    stride: int = 100
    filename: str = "trajectory.h5"


class TrajectoryRecorder:
    """Record simulation states to HDF5 with fixed step stride.

    Data is written to a temporary file during simulation and atomically renamed
    to the final HDF5 path when closed, so the final artifact appears after run.
    """

    def __init__(self, out_dir: str, config: TrajectoryRecorderConfig) -> None:
        self.out_dir = out_dir
        self.config = config
        self.enabled = bool(config.enabled)
        self.stride = max(1, int(config.stride))
        self.filename = str(config.filename)

        self._h5 = None
        self._tmp_path: str | None = None
        self._final_path: str | None = None
        self._frames_written = 0
        self._last_step: int = -1
        self._n_particles: int = 0

    @property
    def frames_written(self) -> int:
        return int(self._frames_written)

    @property
    def output_path(self) -> str | None:
        return self._final_path

    @property
    def last_step(self) -> int:
        return int(self._last_step)

    def start(
        self,
        n_particles: int,
        *,
        Lx: float,
        Ly: float,
        Lz: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return

        try:
            import h5py  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Trajectory recording requires h5py. Install it via: pip install h5py"
            ) from exc

        os.makedirs(self.out_dir, exist_ok=True)
        final_path = os.path.join(self.out_dir, self.filename)
        tmp_path = final_path + ".tmp"

        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        self._h5 = h5py.File(tmp_path, "w")
        self._tmp_path = tmp_path
        self._final_path = final_path
        self._frames_written = 0
        self._last_step = -1
        self._n_particles = int(n_particles)

        h5 = self._h5
        h5.attrs["format"] = "n_bodys_collision_trajectory_v1"
        h5.attrs["Lx"] = float(Lx)
        h5.attrs["Ly"] = float(Ly)
        h5.attrs["Lz"] = float(Lz)
        h5.attrs["stride"] = int(self.stride)
        h5.attrs["n_particles"] = int(n_particles)

        if metadata:
            meta_grp = h5.create_group("meta")
            for k, v in metadata.items():
                try:
                    meta_grp.attrs[str(k)] = v
                except Exception:
                    meta_grp.attrs[str(k)] = str(v)

        h5.create_dataset("particle_id", data=np.arange(n_particles, dtype=np.int64))
        h5.create_dataset("step", shape=(0,), maxshape=(None,), dtype=np.int64)
        h5.create_dataset("time", shape=(0,), maxshape=(None,), dtype=np.float64)
        h5.create_dataset(
            "position",
            shape=(0, n_particles, 3),
            maxshape=(None, n_particles, 3),
            dtype=np.float32,
            chunks=(1, n_particles, 3),
            compression="gzip",
            compression_opts=4,
        )
        h5.create_dataset(
            "velocity",
            shape=(0, n_particles, 3),
            maxshape=(None, n_particles, 3),
            dtype=np.float32,
            chunks=(1, n_particles, 3),
            compression="gzip",
            compression_opts=4,
        )

    def should_record(self, step: int) -> bool:
        if not self.enabled:
            return False
        return (int(step) % self.stride) == 0

    def append(
        self,
        *,
        step: int,
        t: float,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        vx: np.ndarray,
        vy: np.ndarray,
        vz: np.ndarray,
    ) -> None:
        if not self.enabled:
            return
        if self._h5 is None:
            raise RuntimeError("TrajectoryRecorder.append called before start().")

        n = int(x.shape[0])
        if n != self._n_particles:
            raise ValueError("Particle count changed during simulation; trajectory writer expects fixed particle count.")

        k = self._frames_written
        h5 = self._h5

        ds_step = h5["step"]
        ds_time = h5["time"]
        ds_pos = h5["position"]
        ds_vel = h5["velocity"]

        ds_step.resize((k + 1,))
        ds_time.resize((k + 1,))
        ds_pos.resize((k + 1, n, 3))
        ds_vel.resize((k + 1, n, 3))

        ds_step[k] = int(step)
        ds_time[k] = float(t)

        # Write SoA -> AoS without creating large temporary arrays.
        ds_pos[k, :, 0] = x
        ds_pos[k, :, 1] = y
        ds_pos[k, :, 2] = z

        ds_vel[k, :, 0] = vx
        ds_vel[k, :, 1] = vy
        ds_vel[k, :, 2] = vz

        self._frames_written = k + 1
        self._last_step = int(step)

    def close(self, *, runtime_seconds: float | None = None) -> None:
        if not self.enabled:
            return
        if self._h5 is None:
            return

        if runtime_seconds is not None:
            self._h5.attrs["runtime_seconds"] = float(runtime_seconds)

        self._h5.flush()
        self._h5.close()
        self._h5 = None

        if self._tmp_path is not None and self._final_path is not None:
            os.replace(self._tmp_path, self._final_path)
