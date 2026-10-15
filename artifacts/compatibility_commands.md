# 隔离比赛版本 CPU 验证

工作目录：`E:\PycharmCodes\计挑人工智能赛道`。原有 `.venv` 和系统 Python 不作修改；uv、独立 CPython、缓存及新环境位于当前工作区。无 GPU 操作。

参考：[uv 安装与隔离 Python](https://docs.astral.sh/uv/guides/install-python/)、[uv 环境变量](https://docs.astral.sh/uv/reference/environment/)、[PyTorch 官方历史版本配对](https://pytorch.org/get-started/previous-versions/)。

```powershell
# uv 从官方 PyPI 安装到独立目录（不是当前 venv 的 site-packages）。
.\.venv\Scripts\python.exe -m pip install --target tools/uv uv --index-url https://pypi.org/simple

# 首次 wheel 请求长时间未返回；改为 PyPI JSON 提供的同一官方文件 URL，
# 添加缓存刷新参数并通过 pip 的 #sha256=... 机制验证后成功。
.\.venv\Scripts\python.exe -m pip install --target tools/uv --no-deps 'https://files.pythonhosted.org/packages/2a/8e/d23278bb327d2a75dc1d033c81b7fe23bbb28e363e02e7f306abd047924a/uv-0.13.1-py3-none-win_amd64.whl?cachebust=20261015compat#sha256=4f693cd4feb52f08f30474c4612f24be2050472609be65b9960b635f286da284'

$env:UV_PYTHON_INSTALL_DIR = Join-Path (Get-Location).Path 'tools/python'
$env:UV_CACHE_DIR = Join-Path (Get-Location).Path 'tools/uv-cache'
.\tools\uv\bin\uv.exe python install 3.12 --no-bin --no-registry --no-config
.\tools\uv\bin\uv.exe venv .venv-competition --python 3.12 --managed-python --no-config

# 使用 PyTorch 官方 CPU 索引；其余依赖来自 PyPI。
.\tools\uv\bin\uv.exe pip install --python .venv-competition/Scripts/python.exe --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match 'torch==2.10.0+cpu' 'torchvision==0.25.0+cpu' 'numpy>=2.0,<3' 'pandas>=2.2,<4' 'Pillow>=11,<13' 'scikit-learn>=1.6,<2' 'matplotlib>=3.9,<4'

.\.venv-competition\Scripts\python.exe -X utf8 tools/verify_compatibility.py --checkpoint runs/gpu_smoke/best.pt --checkpoint runs/convnext_tiny_320_seed43/best.pt

# 正式权重生成后，仅追加 checkpoint CPU 检查，无需重复单测：
# .\.venv-competition\Scripts\python.exe -X utf8 tools/verify_compatibility.py --skip-tests --checkpoint runs/convnext_tiny_320_seed43/best.pt
```

结果保存到 `artifacts/compatibility.json`，单测明细在 `artifacts/compatibility_unittest.log`。此验证覆盖 Python / PyTorch / torchvision 版本与 checkpoint 加载、CPU 推理，不代表已经验证 Linux 目标容器或 T4 延迟。

2026-10-15 21:22（香港时间）结果：CPython 3.12.15 / torch 2.10.0+cpu / torchvision 0.25.0+cpu；全部 11 项现有单测通过（原目标 9 项，执行时新增了 2 项 pipeline 测试）。DenseNet smoke 与正式 ConvNeXt 权重均成功严格加载，对真实胸片前向输出 `(1, 10)` 且全部有限，`cuda_initialized=false`。
