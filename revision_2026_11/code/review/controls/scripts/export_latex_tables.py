"""Export only verified paired-ablation aggregates to the new SI sources."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "tables/latex"


def value(row, key):
    metric = row[key]
    assert row["n"] == 5 and metric["sd"] is not None
    return f"${metric['mean']:.6f} \\pm {metric['sd']:.6f}$"


def main():
    data = json.loads((ROOT / "tables/aggregates.json").read_text(encoding="utf-8"))
    verification = json.loads((ROOT / "audit/result_validation.json").read_text(encoding="utf-8"))
    assert verification["completed_variant_rows"] == 40
    assert verification["source_regularizer_metrics_recomputed"]
    DEST.mkdir(parents=True, exist_ok=True)
    gate = [r"\begin{table}[!ht]", r"\centering\small",
            r"\caption{Paired Stage I gate controls. Values are mean $\pm$ sample standard deviation over five training seeds. Dimensions are active slots; each listed dimension was selected in all five runs. Both arms have $f_s=0$. Errors are source-standardized derivative RMSE, not trajectory RMSE.}",
            r"\label{tab:paired_gate}", r"\setlength{\tabcolsep}{5pt}",
            r"\begin{tabular}{llccc}", r"\toprule",
            r"System & Gate & Dimensions & Source validation & Target query \\", r"\midrule"]
    for system, title in (("pendulum", "Measured pendulum"), ("quadrotor", "Quadrotor")):
        for variant, label in (("gate_on", "Learned"), ("gate_off", "All active")):
            row = data["gate"][system+"/"+variant]
            dimensions = sorted(set(row["active_dim"]))
            assert len(dimensions) == 1
            gate.append(f"{title} & {label} & {dimensions[0]} & {value(row, 'source_full_nrmse')} & {value(row, 'query_nrmse_after')} \\\\")
    gate += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    regularizer = [r"\begin{table}[!ht]", r"\centering\small",
            r"\caption{Fixed-template symbolic reconstruction controls. Values are mean $\pm$ sample standard deviation over five feedback repetitions from the same teacher and consolidated template within each system. The symbolic expression remains present at $\lambda=0$. Lorenz target observations use a parameter configuration also present among the source tasks.}",
            r"\label{tab:paired_regularizer}", r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{lcccc}", r"\toprule",
            r"System & $\lambda$ & Source: $f_s+f_n$ & Source: $f_s$ & Target: $f_s+f_n$ \\", r"\midrule"]
    for system, title in (("pendulum", "Measured pendulum"), ("lorenz", "Lorenz")):
        for variant, label in (("lambda_1", "1"), ("lambda_0", "0")):
            row = data["separation"][system+"/"+variant]
            regularizer.append(f"{title} & {label} & {value(row, 'source_full_nrmse')} & {value(row, 'source_prior_nrmse')} & {value(row, 'query_nrmse_after')} \\\\")
    regularizer += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    for name, lines in (("gate_ablation.tex", gate), ("regularizer_ablation.tex", regularizer)):
        (DEST/name).write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("Exported two tables from 40 validated model results.")


if __name__ == "__main__":
    main()
