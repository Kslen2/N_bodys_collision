from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any

import numpy as np


_HERE = os.path.abspath(os.path.dirname(__file__))
_PKG_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
DEFAULT_H5_FOLDER = os.path.join(_PKG_ROOT, "data", "N10_2.5e-10dt_10e-5_verlet3_h54")


def _find_h5_file(folder: str) -> str:
    candidate = os.path.join(folder, "trajectory.h5")
    if os.path.isfile(candidate):
        return candidate
    all_h5 = sorted(glob.glob(os.path.join(folder, "*.h5")))
    if not all_h5:
        raise FileNotFoundError(f"No HDF5 file found in folder: {folder}")
    return all_h5[0]


def _analyze_h5(h5_path: str) -> dict[str, Any]:
    try:
        import h5py  # type: ignore
    except Exception as exc:
        raise RuntimeError("This script requires h5py. Install it via: pip install h5py") from exc

    with h5py.File(h5_path, "r") as h5:
        if "time" not in h5 or "position" not in h5 or "velocity" not in h5:
            raise ValueError("Invalid trajectory file format: required datasets are missing.")

        time = h5["time"][:]
        step = h5["step"][:] if "step" in h5 else np.arange(time.size, dtype=np.int64)
        pos = h5["position"]
        vel = h5["velocity"]

        frames = int(time.size)
        if frames <= 0:
            raise ValueError("Trajectory file contains no frames.")

        n_particles = int(pos.shape[1])
        duration = float(time[-1] - time[0]) if frames > 1 else 0.0
        dt_stats = {}
        if frames > 1:
            dt = np.diff(time)
            dt_stats = {
                "dt_mean": float(np.mean(dt)),
                "dt_min": float(np.min(dt)),
                "dt_max": float(np.max(dt)),
            }
        else:
            dt_stats = {"dt_mean": 0.0, "dt_min": 0.0, "dt_max": 0.0}

        mean_speed_series = np.empty(frames, dtype=np.float64)
        max_speed_series = np.empty(frames, dtype=np.float64)
        span_x = np.empty(frames, dtype=np.float64)
        span_y = np.empty(frames, dtype=np.float64)
        span_z = np.empty(frames, dtype=np.float64)

        p0 = pos[0, :, :].astype(np.float64)
        p_last = pos[frames - 1, :, :].astype(np.float64)
        disp = p_last - p0
        rms_displacement = float(np.sqrt(np.mean(np.sum(disp * disp, axis=1))))

        for k in range(frames):
            v = vel[k, :, :].astype(np.float64)
            speed = np.sqrt(v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1] + v[:, 2] * v[:, 2])
            mean_speed_series[k] = float(np.mean(speed))
            max_speed_series[k] = float(np.max(speed))

            pk = pos[k, :, :]
            span_x[k] = float(np.max(pk[:, 0]) - np.min(pk[:, 0]))
            span_y[k] = float(np.max(pk[:, 1]) - np.min(pk[:, 1]))
            span_z[k] = float(np.max(pk[:, 2]) - np.min(pk[:, 2]))

        summary = {
            "file": h5_path,
            "frames": frames,
            "n_particles": n_particles,
            "t_start": float(time[0]),
            "t_end": float(time[-1]),
            "duration": duration,
            "step_start": int(step[0]),
            "step_end": int(step[-1]),
            "rms_displacement": rms_displacement,
            "mean_speed_start": float(mean_speed_series[0]),
            "mean_speed_end": float(mean_speed_series[-1]),
            "max_speed_start": float(max_speed_series[0]),
            "max_speed_end": float(max_speed_series[-1]),
            "span_x_start": float(span_x[0]),
            "span_x_end": float(span_x[-1]),
            "span_y_start": float(span_y[0]),
            "span_y_end": float(span_y[-1]),
            "span_z_start": float(span_z[0]),
            "span_z_end": float(span_z[-1]),
        }
        summary.update(dt_stats)

        return {
            "summary": summary,
            "time": time,
            "mean_speed_series": mean_speed_series,
            "max_speed_series": max_speed_series,
            "span_x": span_x,
            "span_y": span_y,
            "span_z": span_z,
        }


def _save_outputs(folder: str, result: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = result["summary"]
    time = result["time"]

    summary_json = os.path.join(folder, "trajectory_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    summary_txt = os.path.join(folder, "trajectory_summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(time, result["mean_speed_series"], label="mean |v|")
    axes[0].plot(time, result["max_speed_series"], label="max |v|")
    axes[0].set_ylabel("speed (m/s)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(time, result["span_x"], label="span x")
    axes[1].plot(time, result["span_y"], label="span y")
    axes[1].plot(time, result["span_z"], label="span z")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("extent (m)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig_path = os.path.join(folder, "trajectory_analysis.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay/analyze trajectory HDF5 in a simulation output folder")
    parser.add_argument(
        "folder",
        type=str,
        nargs="?",
        default=DEFAULT_H5_FOLDER,
        help="Folder that contains trajectory.h5 (or another .h5 file)",
    )
    args = parser.parse_args()

    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"Folder does not exist: {folder}")

    h5_path = _find_h5_file(folder)
    result = _analyze_h5(h5_path)
    _save_outputs(folder, result)

    summary = result["summary"]
    print("[Replay] HDF5:", h5_path)
    print("[Replay] Frames:", summary["frames"], "Particles:", summary["n_particles"])
    print("[Replay] Time:", f"{summary['t_start']:.3e} -> {summary['t_end']:.3e} s")
    print("[Replay] RMS displacement:", f"{summary['rms_displacement']:.3e} m")
    print("[Replay] Mean speed start/end:", f"{summary['mean_speed_start']:.3e} / {summary['mean_speed_end']:.3e} m/s")
    print("[Replay] Max speed start/end:", f"{summary['max_speed_start']:.3e} / {summary['max_speed_end']:.3e} m/s")
    print("[Replay] Saved:")
    print("  -", os.path.join(folder, "trajectory_summary.txt"))
    print("  -", os.path.join(folder, "trajectory_summary.json"))
    print("  -", os.path.join(folder, "trajectory_analysis.png"))


if __name__ == "__main__":
    main()
