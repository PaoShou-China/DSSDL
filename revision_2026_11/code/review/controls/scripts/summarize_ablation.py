"""Recompute metrics from saved predictions and summarize all completed pairs."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def fmt(values):
    mean = statistics.mean(values)
    return f"{mean:.6f}" if len(values) == 1 else f"{mean:.6f} +/- {statistics.stdev(values):.6f}"


def main():
    records = []
    guards = {}
    separation_details = []
    for path in sorted((ROOT / "experiments").glob("*/*/seed_*/summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["status"] == "completed"
        assert data["protected_inputs_unchanged"]
        guards.update(data["manifest"]["protected_hashes"])
        results = data["results"]
        if path.parents[2].name == "separation":
            separation_details.append(data)
        assert len({v["initialization_hash"] for v in results.values()}) == 1
        refs = []
        for variant, r in results.items():
            with np.load(path.parent / f"{variant}_predictions.npz") as p:
                reference = p["reference"]
                measured = float(np.sqrt(np.mean((p["prediction"].astype(float)-reference)**2)))
                assert abs(measured-r["query_nrmse_after"]) < 2e-7
                refs.append(reference.copy())
                if path.parents[2].name == "separation":
                    if path.parents[1].name == "lorenz":
                        full, prior, truth = p["source_val_prediction"], p["source_val_symbolic"], p["source_val_reference"]
                    else:
                        m = data["manifest"]
                        raw = np.concatenate([np.loadtxt(item["labels"]) for item in m["source_data"]])
                        normalized = ((raw-raw.mean(0))/raw.std(0)).astype(np.float32)
                        ids = np.asarray(m["source_val_indices"])
                        full, prior, truth = p["normalized_full"][ids], p["normalized_prior"][ids], normalized[ids]
                    assert abs(float(np.sqrt(np.mean((full-truth)**2)))-r["val_full_nrmse"]) < 2e-7
                    assert abs(float(np.sqrt(np.mean((prior-truth)**2)))-r["val_prior_nrmse"]) < 2e-7
            assert not r["query_used_for_stopping"]
            assert r["steps"] == 500
            zero_epochs = None
            if path.parents[2].name == "gate":
                history = json.loads((path.parent / f"{variant}_history.json").read_text(encoding="utf-8"))
                zero_epochs = sum(h.get("active_dim") == 0 for h in history["history"])
            records.append(dict(experiment=path.parents[2].name, system=path.parents[1].name,
                                seed=data["manifest"]["seed"], variant=variant,
                                active_dim=r.get("active_dim", len(r.get("active_slots", []))),
                                query_nrmse_after=r["query_nrmse_after"],
                                query_nrmse_before=r["query_nrmse_before"],
                                source_full_nrmse=r.get("source_val_nrmse", r.get("val_full_nrmse")),
                                source_prior_nrmse=r.get("val_prior_nrmse"),
                                participation_ratio=r.get("participation_ratio"),
                                neural_rms=r.get("neural_rms"),
                                symbolic_rms=r.get("symbolic_rms"),
                                symbolic_energy_fraction=r.get("symbolic_energy_fraction"),
                                component_cross_term=r.get("component_cross_term"),
                                zero_gate_epochs=zero_epochs,
                                wall_seconds=r["wall_seconds"], path=str(path)))
        np.testing.assert_array_equal(refs[0], refs[1])
    for path, expected in guards.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    target = ROOT / "tables/paired_results.csv"
    with target.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lines = ["# 本轮已执行的消融结果", "", "日期：2026-10-27。所有数值来自本目录新运行，修订稿以新传NCP_3为底稿。", "",
             "## 1. 门控成对比较", "",
             "指标是标准化导数输出RMSE，不是状态递推误差；target固定500步适配，query不参与选择。每个系统只沿用原有一个target配置，均值和SD反映训练种子差异，不是多个目标系统的泛化统计。", "",
             "| 系统 | 变体 | n | 有效维数（按seed顺序） | query NRMSE，均值±样本SD | source留出NRMSE |",
             "|---|---|---:|---|---|---|"]
    completed = {}
    for system in ("pendulum", "quadrotor"):
        for variant in ("gate_on", "gate_off"):
            rs = sorted([r for r in records if r["experiment"] == "gate" and r["system"] == system and r["variant"] == variant], key=lambda r: r["seed"])
            if not rs:
                continue
            completed[f"{system}_{variant}"] = len(rs)
            lines.append(f"| {system} | {variant} | {len(rs)} | {','.join(str(r['active_dim']) for r in rs)} | {fmt([r['query_nrmse_after'] for r in rs])} | {fmt([r['source_full_nrmse'] for r in rs])} |")
    lines += ["", "同一seed的两臂初始模型权重、数据切分及目标query已验证相同。门控候选维度均为8，不是任务数。此试验仅评估神经阶段的门控选择，没有符号项，不与闭环正则试验合并成一个统一五臂模型。", "",
              "### 配对差值与维数记录", "",
              "下表差值为gate-on减gate-off，正值表示门控的目标误差更高；没有基于query筛选种子。有效维数是活动槽位数，不直接等同于可辨识的物理参数数目。", "",
              "| 系统 | 配对NRMSE差值，均值±样本SD | 门控误差较低的seed数 | gate-on全关epoch总数 |",
              "|---|---|---:|---:|"]
    for system in ("pendulum", "quadrotor"):
        rs = [r for r in records if r["experiment"] == "gate" and r["system"] == system]
        on = {r["seed"]: r for r in rs if r["variant"] == "gate_on"}
        off = {r["seed"]: r for r in rs if r["variant"] == "gate_off"}
        assert on.keys() == off.keys()
        diffs = [on[s]["query_nrmse_after"]-off[s]["query_nrmse_after"] for s in sorted(on)]
        if diffs:
            lines.append(f"| {system} | {fmt(diffs)} | {sum(d < 0 for d in diffs)}/{len(diffs)} | {sum(r['zero_gate_epochs'] for r in on.values())} |")
    lines += ["", "source按预先确定的1-SE规则选取非零门检查点，并非强行把全关门打开。全关历史计数仍保留；不得由最终非零模型推断训练从未全关。", "",
              "## 2. 固定符号模板的重构正则", "",
              "分别固定原teacher和同一PySR整理模板，100轮反馈。每个系统内部两臂协议一致；统计是固定teacher后的重复，不是独立发现种子。", "",
              "| 系统 | 变体 | n | source完整模型NRMSE | source符号单独NRMSE | target适配后NRMSE |",
              "|---|---|---:|---|---|---|"]
    for system in ("pendulum", "lorenz"):
        for variant in ("lambda_1", "lambda_0"):
            rs = [r for r in records if r["experiment"] == "separation" and r["system"] == system and r["variant"] == variant]
            if rs:
                completed[f"regularizer_{system}_{variant}"] = len(rs)
                lines.append(f"| {system} | {variant} | {len(rs)} | {fmt([r['source_full_nrmse'] for r in rs])} | {fmt([r['source_prior_nrmse'] for r in rs])} | {fmt([r['query_nrmse_after'] for r in rs])} |")
    lines += ["", "lambda=0仍保留fs。这是反馈损失的单因素对照，不等于w/o symbol；没有再次自由符号搜索，所以不能据此声称搜索丢失某项。", "",
              "这两类消融复用的是此前已核查的数据和实现，但不是对旧论文全部性能结果的重新计算。门控选择仅用source留出集，target适配按固定预算执行。随机source切分不等同于按完整轨迹留出；该限制对两臂相同。", "",
              "Lorenz沿用18个source任务和原100点support/500点query；目标文件标识也在source中，但观测行不重复。因此其target指标是同参数新观测适配，不是未见参数外推。单摆source随机80/20切分并取反馈最终轮；Lorenz按每文件时间前80%/后20%切分并按source误差选检查点。两系统分列，不混报。", "",
              "### 单摆固定模板与分量核查", "",
              "两臂均从同一个已记录模板开始：alpha/100 = (c0*q+c1)*sin(theta)+(c2*q+c3)*u；第二输出为c4*u，其中u=omega/20。theta是原数据角度；进入网络前的标准化在符号模块中已逆变换，不能把sin(theta)改成sin(标准化角度)。q是各臂学习的任务坐标，不能将下表系数差直接解释成物理参数误差。", "",
              "| seed | 变体 | c0 | c1 | c2 | c3 | c4 | 神经项RMS | 符号能量比例 | 2*mean(fs*fn) |",
              "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for data in separation_details:
        if data["manifest"]["system"] != "measured_pendulum":
            continue
        seed = data["manifest"]["seed"]
        initial = " | ".join(f"{c:.6f}" for c in data["manifest"]["initial_constants"])
        lines.append(f"| {seed} | 同一初值 | {initial} | - | - | - |")
        for name, r in data["results"].items():
            constants = " | ".join(f"{c:.6f}" for c in r["constants"])
            lines.append(f"| {seed} | {name} | {constants} | {r['neural_rms']:.6f} | {r['symbolic_energy_fraction']:.6f} | {r['component_cross_term']:.6f} |")
    lines += ["", "### Lorenz固定模板与共享系数", "",
              "模板：dx=Q1*(y-x)，dy=Q2*x+c1*x*z+c2*y，dz=c3*x*y+Q3*z。Q是从原已发现系数函数重参数化得到；中心与尺度来自source已发现系数，不是真值参数decoder。", "",
              "| seed | 变体 | c1 | c2 | c3 | 神经项RMS | 符号能量比例 | 2*mean(fs*fn) |",
              "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for data in separation_details:
        if data["manifest"]["system"] != "lorenz":
            continue
        seed = data["manifest"]["seed"]
        initial = " | ".join(f"{c:.6f}" for c in data["manifest"]["initial_constants"])
        lines.append(f"| {seed} | 同一初值 | {initial} | - | - | - |")
        for name, r in data["results"].items():
            constants = " | ".join(f"{c:.6f}" for c in r["constants"])
            lines.append(f"| {seed} | {name} | {constants} | {r['neural_rms']:.6f} | {r['symbolic_energy_fraction']:.6f} | {r['component_cross_term']:.6f} |")
    lines += ["", "分量统计均在标准化输出坐标中计算。能量比例定义为mean(fs²)/(mean(fs²)+mean(fn²))，并不包含两项抵消的交叉项，所以只作为辅助信息。不能仅凭该比例宣称fs趋零或符号结构崩溃。", "",
              "## 3. 数据与验证", "", "- 每一对预测都从保存的NPZ独立重算，和JSON指标相符；两臂query逐元素一致。",
              "- 原数据、teacher、原实现文件哈希复查未变；无物理参数标签或decoder参与训练。",
              "- query若与support有完全重复行，已按协议剔除并在manifest记录。",
              "- 原始全精度结果、模型、日志在experiments/；汇总在tables/paired_results.csv。",
              "- 原NCP_3只读；正文、SI与回复在独立revision_2026-10-27_minimal中同步。", ""]
    lines += ["## 4. 当前可以如何回应", "",
              "- 门控：五种子均选出单摆1维、四旋翼3维。单摆的目标误差下降；四旋翼均值略高，五次中有两次更低。支持紧凑任务表示及其精度权衡，不支持所有系统的门控都提高预测精度，也未证明数学意义的最小可辨识维数。",
              "- 符号重构约束：完整模型误差和符号项单独误差必须分别解释；不把固定模板系数优化说成自由搜索丢失某项，也不将Lorenz同参数评估扩写为参数外推。",
              "- 已有w/o symbol和w/o adaptation：可以继续用原图解释对应作用，但本轮门控/正则数据不是同一训练协议的五臂消融。",
              "- 本轮回应围绕组件作用和简洁工作流。只有要进一步声称正则改变发现结构时，才需要两臂同预算PySR重搜索。", ""]
    (ROOT / "03_新增消融结果.md").write_text("\n".join(lines), encoding="utf-8")
    verification = dict(prediction_metrics_recomputed=True, paired_initializations_match=True,
                        query_arrays_match=True, support_only_final_iteration=True,
                        protected_inputs_unchanged=True, source_regularizer_metrics_recomputed=True, completed_runs=completed,
                        completed_variant_rows=len(records))
    aggregates = {}
    for experiment in ("gate", "separation"):
        aggregates[experiment] = {}
        for r in [v for v in records if v["experiment"] == experiment]:
            key = r["system"] + "/" + r["variant"]
            if key in aggregates[experiment]:
                continue
            group = [v for v in records if v["experiment"] == experiment and v["system"] == r["system"] and v["variant"] == r["variant"]]
            result = dict(n=len(group), seeds=[v["seed"] for v in group], active_dim=[v["active_dim"] for v in group])
            for metric in ("source_full_nrmse", "source_prior_nrmse", "query_nrmse_after", "neural_rms", "symbolic_rms", "symbolic_energy_fraction", "component_cross_term"):
                values = [v[metric] for v in group if v[metric] is not None]
                if values:
                    result[metric] = dict(mean=statistics.mean(values), sd=statistics.stdev(values) if len(values)>1 else None)
            aggregates[experiment][key] = result
    (ROOT / "tables/aggregates.json").write_text(json.dumps(aggregates, indent=2), encoding="utf-8")
    (ROOT / "audit/result_validation.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(json.dumps(verification))


if __name__ == "__main__":
    main()
