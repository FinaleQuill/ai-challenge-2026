# 胸部 X 光多标签分类比赛

本工程读取官方 JPG/CSV，以 10 个独立概率进行多标签预测，按 Study 汇总并选择验证 Macro-AUC 最好的权重。原始数据不修改，训练只使用 train，valid 用于选模与聚合比较。

## 当前环境与规则

- 本机：Windows、Python 3.14.2、PyTorch 2.11.0+cu128、torchvision 0.26.0、RTX 5060 Laptop 8GB；现有环境保留。
- 比赛环境：Python 3.12、PyTorch 2.10.0、torchvision 0.25.0、CUDA 12.8、T4 16GB。独立 `.venv-competition` 已验证 Python 3.12.15 / torch 2.10.0+cpu / torchvision 0.25.0+cpu，15 项测试及真实图片权重加载前向均通过；官方 T4 / CUDA 环境的延迟验收仍待实测，记录见 `artifacts/compatibility.json`。
- 已确认可用预训练。基线采用官方 torchvision ImageNet 权重，下载时校验哈希。
- 若首次下载遭遇网络截断/哈希错误，可运行 `python -m cxr.download_weights --model convnext_tiny`；只从官方来源重试，保留长度和 SHA256 检查，不绕过校验。
- 官方尚未明确是否每 Study 提交全部 10 类概率、遗漏类别如何评分、F1 是否固定阈值。内部始终保留完整概率。`submission.csv` 按全概率解释导出，`submission_threshold_0p5.csv` 按阈值示例导出；**上传前仍需确认官方口径**。
- 模型压缩前总大小 ≤500MB、T4 单图端到端 ≤100ms。笔记本测速不能代替 T4 验收。

## 启动

以下命令均在项目根目录的 PowerShell 执行。正式训练会占用 GPU，应暂停游戏；期间保持系统唤醒，但不强制屏幕常亮。

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m cxr.audit --data-root cxr_first/cxr_first --output artifacts/data_audit.json
.\scripts\start_training.ps1
```

后台流程依次执行：CPU 缓存 → 训练 → 按患者 bootstrap 和 Study 聚合比较 → 验证/测试预测 → 本机端到端测速。可以关闭聊天窗口，进程继续运行；请保持电脑通电、开机，不要合盖休眠。

随时运行 `.\scripts\status.ps1` 查看当前阶段、最佳验证成绩、最近训练进度和 GPU 占用。

## GitHub 克隆与模型权重

仓库不提交 `cxr_first/`、`cache/`、`runs/`、原图和 IDE/虚拟环境。队友克隆后把比赛
数据放回 `cxr_first/cxr_first/`，然后执行：

```powershell
git lfs install
git lfs pull
python -m cxr.prepare --data-root cxr_first/cxr_first --cache-root cache/images512
```

`models/convnext_tiny_320_seed43_best.pt` 是正式实验的最佳推理权重，已用 Git LFS
跟踪；如果队友不需要重新训练，可以直接使用根目录的 `submission.csv`，或从
`results/convnext_tiny_320_seed43/test_probabilities.npz` 重新导出。`last.pt` 的
优化器恢复点有 334MB，留在本机，不提交仓库。

队友要从最佳权重继续训练，先确认 `git lfs pull` 已下载模型，然后运行：

```powershell
.\scripts\start_training.ps1 -Config configs/convnext_tiny_phase2.json -SkipPrepare
```

`phase2` 会从 `best.pt` 恢复模型权重，但重新初始化优化器、调度器和 epoch 计数，因此
适合继续微调，不承诺与未中断训练逐位一致。若要精确恢复某次中断，需要自己保留对应
`last.pt` 并改用 `python -m cxr.train --config <config> --resume <last.pt>`。

若希望在终端直接看日志：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u -m cxr.pipeline --config configs/convnext_tiny.json
```

缓存可重入，完整 train/valid/test 共 33,984 张图；缓存最长边 512，CPU 并发 4，原始约 55GB JPG 不复制。预处理保留长宽比、补黑边、RGB / ImageNet 归一化，不使用随机大裁切或翻转。原图推理同样先 LANCZOS 缩至 512，再 BILINEAR 缩至输入尺寸；缓存 JPEG95 带来微小压缩差异。

灰度原图先单通道缩放再转 RGB，避免重复处理三份相同通道；40 张固定真实图片测试与旧 RGB-first 张量逐值相等，CPU 预处理平均约从 50ms 降至 27ms。该优化不改变已有 RGB 缓存的训练路径，详见 `artifacts/preprocessing_probe.json`。

## 暂停与恢复

```powershell
.\scripts\stop_training.ps1
```

训练在下一个优化器更新边界保存 `last.pt` 并停止。确认 `summary.json` 的 `status` 为 `stopped`、GPU 进程结束后即可玩游戏。恢复前删除当前 run 内的单个 `STOP` 文件，再执行：

```powershell
.\scripts\start_training.ps1 -Resume -SkipPrepare
```

恢复要求保持批大小、学习率、epoch 总数等训练配置一致。`last.pt` 包含优化器、调度器、AMP、随机状态和批次进度；`best.pt` 只保留推理需要的模型及配置。默认不是跨 GPU/驱动逐位确定性训练；已验证 CPU 随机增强下中断恢复的一致性。

## 结果位置

首个正式 run：`runs/convnext_tiny_320_seed43/`。本机短测 ConvNeXt-Tiny 约 142 图/秒、DenseNet121 约 30 图/秒（仅模型训练步骤，不含完整数据读取），据此先训练 ConvNeXt；DenseNet 保留作独立对照。

| 文件 | 内容 |
|---|---|
| `pipeline_status.json` | 当前阶段、进程 PID、是否完成或失败 |
| `summary.json` | 训练状态、最佳 epoch、最佳验证分数 |
| `history.jsonl` | 每轮 loss、各类 AUC/F1、学习率和耗时 |
| `best.pt` / `last.pt` | 推理权重 / 训练恢复点 |
| `best_valid_predictions.npz` | 最佳轮逐图概率、标签和身份信息 |
| `outputs/validation_analysis.json` | mean/max/mean-max 聚合比较及患者 bootstrap |
| `outputs/test_probabilities.npz` | 完整测试概率，支持重新导出 |
| `outputs/submission.csv` | 全 10 类长表，完整测试应有 14,140 行 |
| `outputs/submission_threshold_0p5.csv` | 只保留概率 ≥0.5 的另一解释 |
| `outputs/latency.json` | 包含原图读取/解码/预处理的本机 batch=1 延迟 |
| `outputs/experiment_report.md` | 中文实验报告、训练曲线和各类别 AUC 图 |
| `outputs/inference_bundle.zip` | 推理权重、源码、依赖和结果，不含原始数据或优化器 |

标签顺序固定为：1 纵隔心影增大、2 气胸、3 肺部实变、4 肺炎、5 肺水肿、6 心影扩大、7 肺不张、8 肺部阴影、9 胸腔积液、10 无异常发现。

## 独立推理与后续实验

```powershell
.\.venv\Scripts\python.exe -m cxr.predict --checkpoint runs/convnext_tiny_320_seed43/best.pt --split test --aggregation mean --output-dir outputs/manual
.\.venv\Scripts\python.exe -m cxr.analyze --predictions runs/convnext_tiny_320_seed43/best_valid_predictions.npz
.\scripts\start_training.ps1 -Config configs/densenet121.json -SkipPrepare
.\scripts\start_training.ps1 -Config configs/medical_densenet121.json -SkipPrepare
```

避免同时启动两个训练任务。`predict` 可接多个 checkpoint，采用先逐图等权概率融合、再 Study 聚合的明确顺序；对 max，它与先 Study 再模型融合不同，必须用同一管线验证。

医学权重对照使用作者明确标注 CheXpert 的 DenseNet121/224，仅训练新的十类分类头，骨干和 BN 统计冻结。九种异常按名称迁移原分类头，No Finding 单独学习；模型内部还原作者的单通道 [-1024,1024] 输入，不改数据集接口。原旧格式权重通过严格白名单结构提取，正式训练/推理仅加载纯 tensor 检查点；来源、SHA256、映射和许可证见 `artifacts/chexpert_pretraining.json`。部署不依赖 TorchXrayVision 包，也不需要原始旧格式文件。

当前方案为：预训练骨干小学习率、分类头较大学习率、AMP、梯度累积、AdamW、warmup/cosine，按 Study 图片数的倒数加权，使每次检查在训练中的总贡献相同。DenseNet 与 ConvNeXt 分别做对照，后续分辨率、类别权重、BN、ASL、EMA 单独消融。验证集仅 228 位患者，小幅上升不能直接视为稳定改善；当前验证用于模型/聚合选择，分数不是独立测试估计。

实验依据和一手来源见 [docs/research.md](docs/research.md)。
