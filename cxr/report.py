"""Create a human-readable experiment report and exportable validation plots."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cxr import CLASS_NAMES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    args = p.parse_args()
    run = args.run_dir.resolve()
    out = run / "outputs"
    out.mkdir(exist_ok=True, parents=True)
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    history = [json.loads(line) for line in (run / "history.jsonl").read_text(encoding="utf-8").splitlines() if line]
    valid = json.loads((out / "valid_report.json").read_text(encoding="utf-8"))
    metrics = valid["metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    axes[0].plot([r["epoch"] for r in history], [r["train_loss"] for r in history], marker="o")
    axes[0].set(xlabel="Epoch", ylabel="Training BCE", title="Training objective")
    axes[1].plot([r["epoch"] for r in history], [r["macro_auc"] for r in history], marker="o")
    axes[1].set(xlabel="Epoch", ylabel="Study Macro-AUC", title="Validation (mean pooling)")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.savefig(out / "training_curves.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.2, 4.6), layout="constrained")
    ax.barh(CLASS_NAMES, metrics["per_class_auc"], color="#246b83")
    ax.set(xlabel="Validation AUC", xlim=(0.0, 1.0), title=f"Study-level AUC ({valid['aggregation']} pooling)")
    ax.invert_yaxis()
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    fig.savefig(out / "per_class_auc.png", dpi=160)
    plt.close(fig)
    analysis = json.loads((out / "validation_analysis.json").read_text(encoding="utf-8"))
    interval = analysis["patient_bootstrap_mean_pooling"]["macro_auc_percentile_95_interval"]
    latency_path = out / "latency.json"
    latency = json.loads(latency_path.read_text(encoding="utf-8")) if latency_path.exists() else None
    lines = ["# 首轮实验结果", "", f"生成时间：{datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(timespec='seconds')}。",
             "", f"模型：{summary['config']['model']}；输入尺寸：{summary['config']['image_size']}；最佳轮：{summary['best_epoch']}。",
             f"训练状态：{summary['status']}。仅使用官方 train 训练，valid 用于模型与聚合选择。", "",
             f"Study 验证 Macro-AUC：**{metrics['macro_auc']:.6f}**；Macro-F1（0.5）：{metrics['macro_f1']:.6f}。",
             f"聚合方式：{valid['aggregation']}；859 次检查、228 位患者。",
             "验证分数已经参与选模，不代表独立测试成绩，也不能据此推断官方排名。", "",
             "| 类别 | 验证正例检查数 | AUC | F1（0.5） |", "|---|---:|---:|---:|"]
    for i, name in enumerate(CLASS_NAMES):
        lines.append(f"| {i+1} {name} | {metrics['positive_studies'][i]} | {metrics['per_class_auc'][i]:.6f} | {metrics['per_class_f1'][i]:.6f} |")
    lines += ["", "## 聚合对照", "", "| 聚合 | Macro-AUC |", "|---|---:|"]
    for name, result in analysis["pooling_comparison"].items():
        lines.append(f"| {name} | {result['macro_auc']:.6f} |")
    if interval:
        lines += ["", f"mean 聚合的患者 bootstrap 95% 描述区间：[{interval[0]:.6f}, {interval[1]:.6f}]。",
                  "该区间未校正反复调参/模型选择；不是 selected pooling 的独立置信区间。"]
    if latency:
        lines += ["", "## 本机部署检查", "",
                  f"推理权重：{latency['checkpoint_mib']:.2f} MiB；batch=1 端到端中位数 {latency['median_ms']:.2f} ms，P95 {latency['p95_ms']:.2f} ms。",
                  "计入原始 JPG 读取/解码、预处理、GPU 推理和概率回传；不含冷启动。该结果来自本机，尚未在官方 T4 验收。"]
    lines += ["", "## 提交", "", "`submission.csv` 为完整概率长表，应有 14,140 行；另一份文件按 0.5 阈值过滤。",
              "官方缺失类别概率和 F1 阈值约定仍待确认，当前文件不能宣称已经通过官方评分。",
              "", "![Training curves](training_curves.png)", "", "![Per-class AUC](per_class_auc.png)", ""]
    (out / "experiment_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(str(out / "experiment_report.md"), flush=True)


if __name__ == "__main__":
    main()
