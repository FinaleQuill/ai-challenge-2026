# 胸片比赛实验依据与路线

核验日期：2026-10-15。只引用论文原文、作者维护仓库和官方文档；下面的超参数是本项目待验证的起点，不是论文成绩的复现承诺。

执行次序修订：本机 GPU 短测中，DenseNet121/320 约 30 图/秒，ConvNeXt-Tiny/320 约 142 图/秒，后者显存也满足限制。因此首个完整实验改为 ConvNeXt-Tiny；DenseNet121 保留作独立对照。记录见 `artifacts/training_probe.json` 和 `artifacts/convnext_training_probe.json`。这是本机吞吐的工程选择，不能推断哪一个模型的验证 AUC 更高。

## 当前决策

- 采用 **ImageNet 预训练 DenseNet121 + 10 个独立 logits + BCEWithLogitsLoss** 建立基线；随后比较 ConvNeXt-Tiny。官方迁移学习教程区分了冻结特征提取器与全模型微调；本项目训练规模允许先短暂训练分类头，再用较小学习率微调骨干。[PyTorch 迁移学习教程，更新于 2025-01-27](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- 训练可逐图进行，但模型选择、概率聚合和导出必须在 **Study 级** 完成。CheXpert 原论文采用单图模型，并在多视图检查内逐类别取最大概率；这是可比较的强基线，不意味着 max 一定优于 mean。[Irvin 等，AAAI 2019，Model / Training Procedure](https://arxiv.org/html/1901.07031v1)
- 先保留每个 Study 的完整 10 类概率；阈值只用于独立的 F1 报告。等待官方明确缺失类别及阈值规则，不阻塞训练，不擅自把稀疏预测当作完整 AUC 输入。[scikit-learn ROC-AUC 官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)

## 需要逐项验证的选择

| 选择 | 原始证据及限制 | 本项目的验证方式 |
|---|---|---|
| 输入分辨率 | CheXpert 用 DenseNet121 / 320×320；另一项胸片原始研究发现不同疾病受益于不同尺度，不能认为越大越好。[CheXpert，2019](https://arxiv.org/html/1901.07031v1)、[Haque 等，Journal of Medical Imaging 10(4)，2023-08，作者机构页面](https://www.ornl.gov/publication/effect-image-resolution-automated-classification-chest-x-rays)、[论文全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10403240/) | 先 320 建立结果，再比较 384 / 512；使用相同患者划分、聚合规则和训练预算，观察每类 AUC、峰值显存与吞吐。先保持长宽比和肺野完整；插值放大不能恢复已丢失细节。后两条是实施判断。 |
| BCE 与类别权重 | BCEWithLogitsLoss 合并 sigmoid 和 BCE，数值更稳定；`pos_weight` 改变正例贡献及 precision/recall 权衡，官方没有保证 AUC 提升。[PyTorch 2.10 BCE 文档](https://docs.pytorch.org/docs/2.10/generated/torch.nn.BCEWithLogitsLoss.html) | plain BCE 为对照，再比较温和的正例权重。权重仅从训练集统计；不同时引入强过采样，避免无法归因。是否保留由 Study Macro-AUC 决定。 |
| ASL | ASL 通过不同正负样本衰减及负例概率平移处理正负不平衡；论文主要证明自然图像多标签 **mAP** 改善，并非本比赛胸片 Macro-AUC 保证。[Ridnik 等，ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.html) | 在 BCE 流程可靠后单独比较 ASL；先用论文常见配置作为起点，再控制调参次数。关注少数类别排序是否改善，而非只看损失数值。 |
| 小 batch 的归一化 | BN 依赖每次前向的 batch 统计；GN 与 batch 大小无关，原论文显示其小 batch 训练更稳定。ConvNeXt 架构使用 LayerNorm。[PyTorch 2.10 BatchNorm2d](https://docs.pytorch.org/docs/2.10/generated/torch.nn.BatchNorm2d.html)、[Wu / He，ECCV 2018](https://arxiv.org/abs/1803.08494)、[Liu 等，CVPR 2022，§2.6](https://arxiv.org/html/2201.03545v2) | 默认保留预训练架构；若实际 batch 很小，比较冻结 BN 统计与正常更新，或采用原生 LayerNorm 的 ConvNeXt。直接把预训练 BN 替换为 GN 会改变网络，不应当作无损修复。梯度累积扩大优化器的有效 batch，但不合并 BN 的每次前向统计，这是由两者机制得到的推论。 |
| 学习率及 EMA | CheXpert 使用 Adam 1e-4；ConvNeXt 使用 AdamW、warmup / cosine、EMA，但其原始训练数据量及 batch 与本项目不同。PyTorch 提供标准 EMA 实现及 BN 统计更新指引。[CheXpert](https://arxiv.org/html/1901.07031v1)、[ConvNeXt §3.1](https://arxiv.org/html/2201.03545v2)、[PyTorch 2.10 averaged models](https://docs.pytorch.org/docs/2.10/optim.html#weight-averaging-swa-and-ema) | 骨干学习率以约 1e-4 起步，分类头可更高；短 warmup 后 cosine，记录并保存最佳 Study Macro-AUC。EMA 与原权重分别验证；BN buffer 策略必须明确，不能只平均可训练参数而忽略统计量。 |

## 8GB 显存的工程安排

使用自动混合精度；FP16 配合 GradScaler，梯度裁剪前先 unscale；梯度累积时只在完整有效 batch 结束后 step / update。BF16 是否使用由本机支持及短测决定，不能把两种精度的 scaler 处理混同。保留最终不足一个累积周期的数据，并按其实际样本数归一化。[PyTorch 2.10 AMP 官方示例](https://docs.pytorch.org/docs/2.10/notes/amp_examples.html)

先用短测确定真实 batch、峰值显存和一个 epoch 耗时，再排夜间完整训练。完整训练开始前在聊天中通知用户显存占用；不以“8GB 理应能跑”代替实测。模型大小用只含推理权重的实际文件测量；T4 的 100ms 限制最终必须在指定硬件及规定口径下实测，笔记本数据只作初筛。

## 防止把波动当进步

1. 固定官方患者划分；同患者的所有 Study / 图像只属于同一集合。报告每类 AUC、Macro-AUC、正例数及固定 0.5 的 F1，选模使用连续概率，不使用阈值化后的标签。[ROC-AUC 文档](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
2. 验证集有 859 个 Study、228 名患者（本地审计信息）；多个 Study 可能相关，置信区间应按患者整体重采样。比较两个模型时使用相同的患者抽样计算 AUC 差，避免假设 859 次检查完全独立。[Obuchowski，Biometrics 53(2):567–578，1997-06](https://pubmed.ncbi.nlm.nih.gov/9192452/)
3. AUC 衡量排序，校准衡量概率与真实频率的一致性。逐类严格单调变换保持排序，所以阈值调优或单模型温度缩放不能凭空提升 AUC；isotonic 可能因制造并列改变 AUC。融合模型时校准再求均值可能改变排序，需重新验证。[scikit-learn calibration，ranking metrics 段](https://scikit-learn.org/stable/modules/calibration.html#isotonic)
4. 每轮只改变一个关键因素。先 max / mean / mean-logit 聚合消融；再比较分辨率、损失、骨干。只有存在稳定互补时才试少量集成，同时检查 500MB 和延迟预算。max 之外的聚合是本项目实验候选，不是 CheXpert 原论文结论。

## 公开胸片权重的来源核验

允许使用预训练已由用户确认，无需再次询问。基线采用来源清楚的通用 ImageNet 权重，目的是建立可解释的对照。

TorchXRayVision 作者仓库明确提供 CheXpert、MIMIC-CXR 及混合数据权重；其部分权重只有若干输出被训练，不能直接把 18 个输出按位置映射成本比赛 10 类。[作者仓库 Models](https://github.com/mlmed/torchxrayvision#models-demo-notebook) MIMIC-CXR-JPG 官方说明标签来自 Study 级报告，并提供患者 / Study / 视图元数据。[PhysioNet v2.1.0，2024-03-12](https://physionet.org/content/mimic-cxr-jpg/2.1.0/)

本地 CSV 的 ID 已重新编号，仅凭列名和胸片外观不能证明比赛数据属于哪个公开队列，**目前没有证明公开权重与本比赛验证/测试集重合的证据**。但若后续查明来自相同队列，必须核对预训练患者及划分，避免把见过验证患者的权重当作干净本地验证。记录下载 URL、权重哈希、预训练数据和标签映射；来源不明的胸片权重不作为首个基线。

## 实验顺序

1. 数据完整性 / Study 标签一致性 / 患者隔离检查 → 小样本拟合与推理冒烟测试。
2. DenseNet121 / ImageNet / 320 / BCE → 持久化逐图、逐 Study 概率及配置，跑通完整导出。
3. 无需重训的 Study 聚合消融 → 384 / 512 对照 → 温和加权 BCE / ASL。
4. ConvNeXt-Tiny 对照 → 不同随机种子及 EMA → 如有稳定增益，试预算内互补融合。
5. 根据官方最终格式导出；在比赛兼容环境和 T4 验证体积 / 延迟。提交规则尚未答复时保留完整概率，不丢弃低于阈值的类别。

## 补充：CheXpert 单来源医学预训练对照（2026-10-15）

作者官方模型表、README 与论文命名说明确认 `densenet121-res224-chex` 对应 CheXpert，区别于 MIMIC 与 `all` 混合权重。[官方模型定义](https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/models.py)、[TorchXRayVision 论文，2021](https://arxiv.org/abs/2111.00595) 官方 GitHub 资产已下载，28,381,998 字节；SHA256 为 `1a914ae802170187a4852e594708206c12468eeeefd66aa4e8e26141649d61a4`，与作者 Hugging Face 镜像的 LFS 对象一致。该镜像 model card 的 datasets 标签笼统列出混合数据，与其自身 config 和官方模型表不一致；本项目记录此矛盾，不据该标签认定实际权重为混合模型。[官方资产](https://github.com/mlmed/torchxrayvision/releases/download/v1/chex-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt)、[镜像配置](https://huggingface.co/torchxrayvision/densenet121-res224-chex/blob/main/config.json)

模型是单通道 DenseNet121，输入 224，按 `(2 * (pixel / 255) - 1) * 1024` 归一化；18 个输出中有 11 项有效，本比赛的 9 种异常均覆盖，No Finding 需独立初始化。可复用骨干和正确映射的 9 行分类器，冻结骨干做 4 轮短对照；原模型的 `op_threshs` 概率校准不迁入新 BCE logits 训练头。该选择有领域特征复用依据，是否优于 ImageNet 仍以 Study Macro-AUC 实测判断。[官方归一化与 forward 定义](https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/utils.py)、[分类器定义](https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/models.py)

原文件为旧完整模块 pickle。转换器仅允许静态核验的 16 个精确全局，将作者自定义类替换为空壳容器；未导入或执行下载源码，未调用原模型 forward。使用受限解码器处理旧模块 persistent ID 后，仅保存 tensor state_dict 与基础类型来源信息；纯文件已通过 `weights_only=True` 重读和 torchvision 单通道 18 类 DenseNet121 的 `strict=True` 加载核验（727 个键）。后续训练和推理直接使用纯权重，无需 xrv 包或旧 pickle。完整证据、哈希及白名单见 `artifacts/chexpert_pretraining.json`，转换过程见 `tools/convert_xrv_chex.py`。

作者仓库及镜像将许可证标为 Apache-2.0；原文已保存到 `docs/third_party/TorchXrayVision-LICENSE.txt`，核验时仓库根目录未提供 NOTICE 文件。[上游 LICENSE](https://github.com/mlmed/torchxrayvision/blob/master/LICENSE)、[作者镜像许可标注](https://huggingface.co/torchxrayvision/densenet121-res224-chex/blob/main/README.md)
