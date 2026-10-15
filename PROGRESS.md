# 进度交接（2026-10-15）

用户最新要求：完成当前 ConvNeXt 实验后停止扩展，保存进度，交付 CSV 供平台试测；等平台反馈后继续。不要自动启动医学权重或其他训练。

GitHub：当前工作区将直接覆盖 `FinaleQuill/ai-challenge-2026` 的旧 main。提交内容包括代码、配置、测试、文档、小型报告、提交 CSV、测试概率和 `models/convnext_tiny_320_seed43_best.pt`（Git LFS）。不提交 `cxr_first/`、`cache/`、`runs/`、`last.pt`、原始图片或虚拟环境。

队友续训入口：`configs/convnext_tiny_phase2.json` 从 LFS 中的 best.pt 热启动，重新初始化优化器/调度器；CLI 和单元测试均已验证。需要精确断点恢复时，使用自己的 `last.pt` 加 `--resume`。

本次已结束：第10轮触发早停，最佳第6轮，Study Macro-AUC=0.8011423464，mean聚合。pipeline已complete，根目录`submission.csv`已导出并通过14140行/1414 Study/每Study10类/ID全覆盖/有限概率范围检查。完整模型、优化器恢复点、日志及推理ZIP均已保存。后续无训练在运行。

## 已完成

- 官方数据审计通过：train 30344 图/18513 Study，valid 1422 图/859 Study，test 2218 图/1414 Study；患者 ID 无交集。原图不改。
- 全部 33984 张图片已生成最长边512缓存；训练按患者隔离、Study评估，多标签10 logits/BCE。
- 已完成训练、断点恢复、预测、提交校验、聚合比较、患者bootstrap、推理打包和报告流程。
- 现有GPU环境15项测试通过；隔离Python3.12/torch2.10/vision0.25 CPU环境15项测试及最终最佳权重兼容验证通过，T4实际延迟尚未验收。
- 原图灰度先缩放再RGB的优化，在1422张valid的320输入上张量全部精确相同。
- 研究、官方权重来源/哈希、下载截断修复、许可证均已记录在docs和artifacts。

## 当前实验与文件

正式实验：`runs/convnext_tiny_320_seed43`；配置：`configs/convnext_tiny.json`。

- `pipeline_status.json`：当前阶段/最终完成状态。
- `summary.json`、`history.jsonl`：实际最佳轮和验证成绩，以这两个文件为准。
- `best.pt`：推理权重；`last.pt`：完整恢复点。
- `outputs/submission.csv`：每Study全部10概率的长表，完整应有14140行。
- `outputs/submission_threshold_0p5.csv`：按0.5过滤的备选格式。
- `outputs/experiment_report.md`、`outputs/inference_bundle.zip`：报告及可复现推理包。

查询状态：`./scripts/status.ps1`。需要暂停时运行`./scripts/stop_training.ps1`，等待pipeline状态stopped后再玩游戏。已结束的训练无需再启动。

## 待平台反馈

允许预训练已确认。官方仍未回答：AUC是否要求每Study全部10类概率、遗漏类别如何计分、F1是否固定0.5。先提供完整概率版本供平台试测，保留阈值版；不宣称已通过官方评分。

## 下次可以继续

- 先查看平台评分/报错，再决定提交格式与下一项实验。
- CheXpert-only医学权重已核验并安全转换，`cxr/medical.py`、`configs/medical_densenet121.json`已通过CPU测试，**尚未进行比赛训练**。
- 医学对照是224输入、冻结骨干/BN、4轮分类头；9种异常迁移原头，NoFinding单独学习；不依赖原始pickle或TorchXrayVision包。
- 不要自动重训当前ConvNeXt或启动候选实验；用户希望先在平台验证效果。
