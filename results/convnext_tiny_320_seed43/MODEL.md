# ConvNeXt-Tiny/320 结果卡

这是本地正式实验 `runs/convnext_tiny_320_seed43` 的只读结果副本。远程仓库不提交
`runs/`、`cache/`、`cxr_first/` 和 `last.pt`，因此这里保存可直接检查的预测、报告和
配置摘要。

| 项目 | 值 |
|---|---|
| 模型 | torchvision `convnext_tiny`，ImageNet-1K 预训练 |
| 输入 | 320×320，等比缩放补黑边，ImageNet 归一化 |
| 输出 | 10 个独立 logits，推理时 sigmoid |
| 训练 | BCEWithLogitsLoss，AdamW，AMP，batch 12，梯度累积 2 |
| Study 聚合 | mean 概率 |
| 最佳 epoch | 6 / 10，早停 |
| 验证 Macro-AUC | 0.8011423464 |
| 验证 Macro-F1（0.5） | 0.3559638323 |
| 患者 bootstrap 95% | 0.761219～0.839635 |
| 推理权重 | `models/convnext_tiny_320_seed43_best.pt`，Git LFS |
| 权重 SHA256 | `1d3da90179423741c6e2b327f3f68dc7631056332394d646df0476220daa3313` |

`submission.csv` 为每 Study 全部 10 类概率的长表，共 14,140 行；
`submission_threshold_0p5.csv` 是按 0.5 过滤的备选格式。官方尚未明确评分时
缺失类别和 F1 阈值的处理，提交前应先确认。本机 batch=1 端到端中位数约
32.9 ms、P95 约 40.3 ms，官方 T4 尚未实测。

完整训练曲线、各类别 AUC 和逐图/逐 Study 概率分别见
`training_curves.png`、`per_class_auc.png`、`valid_probabilities.npz` 和
`test_probabilities.npz`。
