# Model weights

`convnext_tiny_320_seed43_best.pt` is the best validation checkpoint from
`runs/convnext_tiny_320_seed43` (epoch 6). It is stored with Git LFS because
the file is larger than GitHub's normal 100 MB limit.

```powershell
git lfs install
git lfs pull
```

The checkpoint contains only the inference model `state_dict` and its config.
It does not contain the competition images, optimizer state, or the 334 MB
`last.pt` resume checkpoint. Use `cxr.predict` as shown in the root README.

To continue training from these weights, use the warm-start config:

```powershell
.\scripts\start_training.ps1 -Config configs/convnext_tiny_phase2.json -SkipPrepare
```
