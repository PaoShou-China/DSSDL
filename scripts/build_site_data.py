"""Export small, traceable website plots from the archived trajectory records."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "revision_2026_11"


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    metrics = rows(ARCHIVE / "results/wasserstein/reference.csv")
    result = {}
    for system, dims in (("lorenz", (0, 2)), ("fhn", (0, 1))):
        methods = {}
        for method in ("dssdl", "no_symbol", "sindy"):
            source = ARCHIVE / "code/review/experiments/wasserstein/data" / system / f"{method}.csv"
            duration = 3.5 if system == "lorenz" else 5.0
            data = [row for row in rows(source) if float(row["time"]) <= duration + 1e-9]
            methods[method] = [[round(float(row[f"pred_dim{d}"]), 6) for d in dims] for row in data]
        reference = [[round(float(row[f"true_dim{d}"]), 6) for d in dims] for row in data]
        result[system] = {
            "reference": reference,
            "methods": methods,
            "duration": float(data[-1]["time"]),
            "axes": ["X", "Z"] if system == "lorenz" else ["v", "w"],
            "wasserstein": [
                {
                    "method": row["label"],
                    "value": float(row["sliced_wasserstein_1"])
                    if row["sliced_wasserstein_1"]
                    else None,
                }
                for row in metrics
                if row["system"] == system
            ],
            "source": f"revision_2026_11/code/review/experiments/wasserstein/data/{system}",
        }
    destination = ROOT / "docs/assets/results.js"
    destination.write_text(
        "window.DSSDL_RESULTS = " + json.dumps(result, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    print("Exported original FHN/Lorenz trajectory records to", destination)


if __name__ == "__main__":
    main()
