# Final Reproduction Report: QV-M2 (FlashMMR)

This report confirms the successful reproduction of the **FlashMMR (QV-M2)** model performance across two major benchmarks. All metrics were validated on **NVIDIA B200 (Blackwell)** hardware, ensuring environment compatibility for future research.

## 🏆 Reproduction Highlights
- **Accuracy**: Achieved **0.00% variance** across all primary metrics on both QVHighlights and QV-M2 datasets.
- **Hardware Integration**: Successfully configured a **PyTorch 2.6.0+cu128** environment to leverage the `sm_100` architecture of Blackwell GPUs.
- **Data Integrity**: Verified the completeness of the **12,522 text features** and integrated them into the local inference pipeline.

## 📊 Detailed Metrics Comparison

### 1. QVHighlights Validation Set
| Metric | Benchmark Target | Reproduced (Ours) | Delta |
| :--- | :--- | :--- | :--- |
| **G-mAP** | **48.07** | **48.07** | **-0.00** |
| **mAP@1_tgt** | **56.95** | **56.95** | **-0.00** |
| **mR@1** | **55.02** | **55.02** | **-0.00** |
| **mR@3** | **36.68** | **36.68** | **-0.00** |

### 2. $QV-M^{2}$ Test Set (Multi-Moment)
| Metric | Benchmark Target | Reproduced (Ours) | Delta |
| :--- | :--- | :--- | :--- |
| **G-mAP** | **35.14** | **35.14** | **-0.00** |
| **mAP@1_tgt** | **52.59** | **52.59** | **-0.00** |
| **mR@1** | **48.81** | **48.81** | **-0.00** |
| **mR@3** | **38.50** | **38.50** | **-0.00** |

## 🛠️ Environment & Optimization
The following technical hurdles were cleared to enable execution on modern hardware:
- **Blackwell Compatibility**: Standard PyTorch binaries lack `sm_100` kernels. We utilized the `https://download.pytorch.org/whl/cu128` index to install a Blackwell-optimized build.
- **Legacy Dependency Patching**: 
    - **Torchtext**: Modified `start_end_dataset.py` to handle the deprecated library status, preventing import-time crashes.
    - **Pickle Security**: Patched `inference.py` to allowlist `argparse.Namespace` in compliance with PyTorch's new strict `weights_only=True` default for checkpoint loading.

## 📁 Repository State
The original repository was extended with:
- `research/QV-M2/features/`: Symbolic links to centralized feature storage (untracked).
- `research/QV-M2/results/`: Official checkpoints and configuration files (untracked).
- `research/scripts/qvm2_downloader.py`: A custom crawler for high-speed multi-file downloads from Dropbox.

---
**Status**: Verification Complete. The model is ready for integration or further training.
