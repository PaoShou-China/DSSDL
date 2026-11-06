"""Recompute fixed-parameter SW1 from frozen original trajectory CSVs only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import wasserstein_distance


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[2]
METHODS = {
    "dssdl": ("final_data_symbol", "DSSDL", "#403990", "--"),
    "no_symbol": ("final_data", "w/o symbol", "#1f65a9", "-"),
    "sindy": ("final_data_SINDy", "SINDy", "#f2b63d", "-."),
    "gplearn": ("final_data_GPlearn", "GPLearn", "#e63135", "-."),
    "reptile": ("final_data_Reptile", "Reptile", "#00a5b5", ":"),
    "cavia": ("final_data_CAVIA", "CAVIA", "#35a535", "-."),
}
SYSTEMS = {
    "fhn": {"folder": "FHN", "file": "[0.85, 1.05, 0.85].csv",
            "duration_s": 5.0, "initial": [-1.0, 1.0], "parameters": [0.85, 1.05, 0.85]},
    "lorenz": {"folder": "lorenz", "file": "[9.9, 29.5, 2.9].csv",
               "duration_s": 3.5, "initial": [8.0, -8.0, 27.0], "parameters": [9.9, 29.5, 2.9]},
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_sources():
    entries = []
    for system, cfg in SYSTEMS.items():
        for method, (folder, _, _, _) in METHODS.items():
            source = WORKSPACE / "draw25" / folder / "trajectories_data" / cfg["folder"] / cfg["file"]
            dest = ROOT / "data" / system / f"{method}.csv"
            expected = digest(source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                if digest(dest) != expected:
                    raise RuntimeError(f"Refusing to replace a different frozen input: {dest}")
            else:
                shutil.copy2(source, dest)
            assert digest(dest) == expected == digest(source)
            entries.append({"system": system, "method": method,
                            "source": str(source), "snapshot": dest.relative_to(ROOT).as_posix(),
                            "sha256": expected})
    manifest = ROOT / "data" / "source_manifest.json"
    payload = {"sources": entries, "scope": "Original trajectory exports, not new training or simulation."}
    if manifest.exists():
        assert json.loads(manifest.read_text(encoding="utf-8")) == payload
    else:
        manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def configure_style():
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9., "axes.labelsize": 10.,
        "axes.titlesize": 10.5, "legend.fontsize": 8.4,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.linewidth": .9,
        "xtick.major.width": .8, "ytick.major.width": .8,
        "xtick.major.size": 3.2, "ytick.major.size": 3.2,
        "lines.linewidth": 1.35, "savefig.dpi": 600, "pdf.fonttype": 42,
        "svg.fonttype": "none", "path.simplify": False, "axes.axisbelow": True,
    })


def save_figure(fig, stem):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        labels = [ax.xaxis.label, ax.yaxis.label, ax.title] + ax.get_xticklabels() + ax.get_yticklabels()
        for artist in labels:
            if not artist.get_visible() or not artist.get_text():
                continue
            box = artist.get_window_extent(renderer)
            # Log tick locators include labels outside the displayed interval.
            if artist in ax.get_yticklabels() and not min(ax.get_ylim()) <= artist.get_position()[1] <= max(ax.get_ylim()):
                continue
            if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1:
                raise RuntimeError(f"Clipped figure text: {artist.get_text()}")
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=600)
    plt.close(fig)


def load_and_compute(manifest):
    rows, curves, checks = [], {}, []
    for system, cfg in SYSTEMS.items():
        dim = len(cfg["initial"])
        expected_time = np.arange(round(cfg["duration_s"] / .01) + 1) * .01
        directions = np.random.RandomState(20260720).normal(size=(128, dim))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        np.save(ROOT / "tables" / f"{system}_projection_directions.npy", directions)
        reference = None
        curves[system] = {}
        for method in METHODS:
            entry = next(e for e in manifest["sources"] if e["system"] == system and e["method"] == method)
            path = ROOT / entry["snapshot"]
            assert digest(path) == entry["sha256"]
            frame = pd.read_csv(path)
            selected = frame[frame.time <= cfg["duration_s"] + 1e-9]
            t = selected.time.to_numpy(dtype=float)
            truth = selected[[f"true_dim{i}" for i in range(dim)]].to_numpy(dtype=float)
            prediction = selected[[f"pred_dim{i}" for i in range(dim)]].to_numpy(dtype=float)
            np.testing.assert_allclose(t, expected_time[:len(t)], rtol=0, atol=1e-12)
            np.testing.assert_array_equal(truth[0], cfg["initial"])
            np.testing.assert_array_equal(prediction[0], cfg["initial"])
            if reference is None:
                reference = truth.copy()
            else:
                np.testing.assert_allclose(truth, reference[:len(t)], rtol=0, atol=1e-12)
            complete = len(t) == len(expected_time) and np.isfinite(prediction).all()
            status = "complete" if complete else "incomplete_diverged"
            value, trajectory_rmse = np.nan, np.nan
            pointwise = np.sqrt(np.mean((prediction - truth) ** 2, axis=1))
            prefix = np.sqrt(np.cumsum(np.sum((prediction - truth) ** 2, axis=1)) / np.arange(1, len(t) + 1))
            if complete:
                distances = [wasserstein_distance(truth @ direction, prediction @ direction) for direction in directions]
                value = float(np.mean(distances))
                sorted_check = np.mean(np.abs(np.sort(truth @ directions.T, axis=0) - np.sort(prediction @ directions.T, axis=0)))
                np.testing.assert_allclose(value, sorted_check, rtol=1e-12, atol=1e-14)
                trajectory_rmse = float(np.sqrt(np.mean((prediction - truth) ** 2)))
            curves[system][method] = (t, pointwise, prefix)
            checks.append({"system": system, "method": method, "snapshot_hash_ok": True,
                           "reference_agrees": True, "complete": bool(complete)})
            rows.append({"system": system, "method": method, "label": METHODS[method][1],
                         "sliced_wasserstein_1": value, "trajectory_rmse": trajectory_rmse,
                         "status": status, "samples_available": len(t), "samples_required": len(expected_time),
                         "requested_end_s": cfg["duration_s"], "available_end_s": float(t[-1]),
                         "prefix_l2_rmse_at_available_end": float(prefix[-1]),
                         "source": entry["source"], "snapshot": entry["snapshot"], "sha256": entry["sha256"]})
        pd.DataFrame({"time": expected_time, **{
            METHODS[m][1]: pd.Series(values[2] if system == "fhn" else values[1])
            for m, values in curves[system].items()
        }}).to_csv(ROOT / "tables" / f"{system}_original_rmse_curves.csv", index=False)
    return pd.DataFrame(rows), curves, checks


def w_panel(system, table):
    fig = plt.figure(figsize=(4.05, 2.60), facecolor="white", dpi=600)
    ax = fig.add_axes((.15, .21, .81, .555))
    rows = table[table.system == system]
    valid = rows[rows.status == "complete"].sliced_wasserstein_1
    low, high = float(valid.min()), float(valid.max())
    logarithmic = high / low > 20
    bottom = max(low / 2, 1e-7) if logarithmic else 0.
    ax.set_ylim(bottom, high * (2.8 if logarithmic else 1.28))
    if logarithmic:
        ax.set_yscale("log")
    for i, (_, row) in enumerate(rows.iterrows()):
        color = METHODS[row.method][2]
        if row.status != "complete":
            ax.scatter(i, .65, marker="x", s=28, color=color, transform=ax.get_xaxis_transform(), zorder=4)
            ax.text(i, .77, "Failed", ha="center", va="center", fontsize=7.7,
                    transform=ax.get_xaxis_transform(), color=color)
        else:
            ax.bar(i, row.sliced_wasserstein_1 - bottom, bottom=bottom, width=.63,
                   color=color, edgecolor="black", linewidth=.55, zorder=3)
    ax.set_xlim(-.65, len(METHODS) - .35)
    ax.set_xticks(range(len(METHODS)), [v[1].replace("w/o symbol", "w/o\nsymbol") for v in METHODS.values()])
    ax.tick_params(axis="x", labelsize=7.8)
    ax.set_ylabel("Sliced Wasserstein")
    ax.set_title("Fixed parameters", fontsize=10, pad=8)
    ax.grid(axis="y", linestyle="--", linewidth=.55, color="#b8b8b8", alpha=.62)
    save_figure(fig, ROOT / "figures" / system / "01_fixed_wasserstein_original")


def fhn_check(curves):
    fig, ax = plt.subplots(figsize=(5.4, 4.4), dpi=600)
    fig.subplots_adjust(left=.13, right=.97, bottom=.14, top=.97)
    for method in ("dssdl", "no_symbol", "sindy", "reptile", "cavia", "gplearn"):
        t, _, prefix = curves["fhn"][method]
        _, label, color, style = METHODS[method]
        ax.plot(t, prefix, label=label, color=color, linestyle=style, linewidth=1.35)
    ax.set(xlim=(0, 5), ylim=(0, .258), xlabel="Time (s)", ylabel="RMSE")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_yticks(np.arange(0, .251, .05))
    ax.grid(True, linestyle="--", linewidth=.55, color="#b8b8b8", alpha=.62)
    ax.legend(frameon=False, ncol=3, loc="center", fontsize=8.4)
    save_figure(fig, ROOT / "figures" / "audit" / "fhn_g_from_original_csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-from-workspace", action="store_true")
    args = parser.parse_args()
    if args.snapshot_from_workspace:
        snapshot_sources()
    manifest = json.loads((ROOT / "data" / "source_manifest.json").read_text(encoding="utf-8"))
    (ROOT / "tables").mkdir(exist_ok=True)
    configure_style()
    table, curves, checks = load_and_compute(manifest)
    table.to_csv(ROOT / "tables" / "original_trajectory_wasserstein.csv", index=False)
    for system in SYSTEMS:
        w_panel(system, table)
    fhn_check(curves)
    for entry in manifest["sources"]:
        assert digest(ROOT / entry["snapshot"]) == entry["sha256"]
    original_unchanged = all(Path(e["source"]).exists() and digest(Path(e["source"])) == e["sha256"] for e in manifest["sources"])
    report = {"input_count": len(checks), "checks": checks, "original_sources_currently_match": original_unchanged,
              "training_performed": False, "simulation_performed": False,
              "standardization": False, "projection_count": 128, "projection_seed": 20260720,
              "single_stored_trajectory_per_method": True, "error_bars": False,
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "pandas": pd.__version__, "scipy": scipy.__version__, "matplotlib": mpl.__version__}}
    (ROOT / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(table[["system", "label", "sliced_wasserstein_1", "status"]].to_string(index=False))
    print("Original source files unchanged:", original_unchanged)


if __name__ == "__main__":
    main()
