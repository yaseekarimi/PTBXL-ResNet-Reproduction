# PTB-XL ResNet1d_Wang Benchmark Reproduction
Phase 0 – M2

This repository documents the reproduction and diagnostic analysis of the resnet1d_wang benchmark reported by Strodthoff et al. (2021) on the PTB-XL dataset.

The objective was to reproduce the published benchmark under comparable experimental conditions before introducing modifications for later stages of the PhD research.

The two benchmark tasks investigated were:

- **exp0 (all)**: multi-label classification of 71 SCP statements
- **exp3 (rhythm)**: multi-label classification of 12 rhythm statements

Published macro-AUROC reference values:

| Experiment | Task | Published AUROC |
|-----------|------|----------------|
| exp0 | All SCP statements (71 classes) | 0.919 |
| exp3 | Rhythm statements (12 classes) | 0.946 |

---

## Reference Implementation

The reproduction was based on the paper and official source repository:

Strodthoff, N., Wagner, P., Schaeffter, T. and Samek, W. (2021). Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL. *IEEE Journal of Biomedical and Health Informatics*, 25(5), 1519–1528.

DOI: [10.1109/JBHI.2020.3022989](https://doi.org/10.1109/JBHI.2020.3022989)

Official implementation: https://github.com/helme/ecg_ptbxl_benchmarking

The following source files from the official implementation were particularly important during the source audit:

- `fastai_configs.py`
- `fastai_model.py`
- `basic_conv1d.py`
- `timeseries_utils.py`
- `scp_experiment.py`
- `utils.py`
- `ecg_env.yml`

The original implementation used FastAI v1. The reproduction in this repository was implemented in modern PyTorch while progressively auditing the implementation against the original source code.

---

## Dataset

**Dataset:** PTB-XL

**Reference:**

Wagner, P. et al. (2020). PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data*, 7, 154.

DOI: [10.1038/s41597-020-0495-6](https://doi.org/10.1038/s41597-020-0495-6)

For the final source-audited reproduction:

- PTB-XL version: **v1.0.1**
- Records: 21,837
- Sampling rate: 100 Hz
- Signal source: `filename_lr` / `records100/`
- ECG duration: 10 seconds
- Leads: 12

PTB-XL data are not included in this repository.

### Why PTB-XL v1.0.1 Was Downloaded Again

Earlier experiments were performed using PTB-XL v1.0.3, which contained 21,799 available records in the local experimental pipeline.

During investigation of the remaining reproduction gap, the dataset version was audited against the benchmark setup. PTB-XL v1.0.1 contained 21,837 records, resulting in a difference of 38 records.

The missing records were distributed across the official folds, including validation fold 9 and test fold 10. Differences were also observed in some rhythm-label counts.

Therefore, the complete 100-Hz PTB-XL v1.0.1 dataset was downloaded and the experiments were repeated without changing the model.

---

## Data Split

The official PTB-XL stratified fold protocol was used:

- Folds 1–8: Training
- Fold 9: Validation
- Fold 10: Test

For PTB-XL v1.0.1:

| Split | exp0 | exp3 |
|-------|------|------|
| Training | 17,441 | 16,854 |
| Validation | 2,193 | 2,109 |
| Test | 2,203 | 2,103 |

---

## Signal Preprocessing

The reproduction uses the 100-Hz PTB-XL signals.

No additional high-pass, notch, or band-pass filtering was introduced because the benchmark-specific preprocessing in the reference implementation uses standardisation rather than additional ECG filtering.

The preprocessing pipeline was separately verified before the benchmark experiments.

---

## Reproduction Development

The reproduction was developed through several diagnostic stages.

### Initial implementation

The initial implementation established the main benchmark framework:

- PTB-XL at 100 Hz
- 12-lead ECG
- Multi-label classification
- BCE loss
- Folds 1–8 / 9 / 10
- Macro-AUROC evaluation
- ResNet1d_Wang architecture

Five independent GPU runs were performed to investigate run-to-run stability.

For exp0, three of the five runs were within the defined 0.01 reproduction criterion. The mean test AUROC was approximately 0.9109.

For exp3, a larger and persistent difference from the published benchmark was observed.

### Exploratory learning-rate experiment

An exploratory run using max_lr=0.003 was performed to investigate whether the learning-rate schedule explained the exp3 discrepancy.

The experiment did not resolve the exp3 gap and was treated as a diagnostic experiment rather than the final reproduction.

### Source-code audit and v2

The original Strodthoff source code was then examined in greater detail.

Important differences identified included:

- 250-sample random training crops
- overlapping validation/test crops
- maximum aggregation across crop predictions
- AdaptiveConcatPool1d
- hidden classification head
- head dropout
- peak learning rate of 0.01
- weight decay of 0.01
- custom weight initialisation

These source-supported corrections were incorporated into the subsequent implementation.

### v2–v5 diagnostic stages

Further experiments investigated:

- 50 versus 100 training epochs
- FastAI-style OneCycle behaviour
- beta1/momentum scheduling
- per-class exp3 AUROC
- PTB-XL dataset-version differences
- PTB-XL v1.0.1 reproduction

These experiments progressively reduced uncertainty about the remaining difference from the published benchmark.

### v6 → v7 source audit

The final v6-to-v7 transition was based on a direct audit of the original source implementation.

The main source-supported corrections were:

- Exact weight initialisation behaviour, including bias and BatchNorm initialisation.
- Training crop generation changed to reproduce the Python `random.randint()` behaviour used in `TimeseriesDatasetCrops`.
- Final-epoch evaluation retained as the source-faithful evaluation behaviour. Both the best-validation checkpoint and the final-epoch checkpoint are evaluated and reported side by side (see M2 Closure below). The primary M2 checkpoint criterion remains subject to supervisor confirmation.

No new architecture, loss function, crop size, learning rate, epoch count, or other performance-oriented modification was introduced in v7.

---

## Final v7 Configuration

| Parameter | Final configuration |
|-----------|-------------------|
| Script | `resnet_exp0_exp3_v7_source_audited_v101.py` |
| Architecture | ResNet1d_Wang |
| Dataset | PTB-XL v1.0.1 |
| Sampling rate | 100 Hz |
| Input ECG | 12 leads × 10 s |
| Training crop | 250 samples |
| Val/Test crop stride | 125 samples |
| Crop aggregation | Maximum |
| Loss | BCE |
| Batch size | 128 |
| Epochs | 50 |
| Maximum LR | 0.01 |
| Weight decay | 0.01 |
| LR schedule | One-cycle (FastAI v1) |
| Filtering | None |
| Train folds | 1–8 |
| Validation fold | 9 |
| Test fold | 10 |
| Evaluation | Macro-AUROC |
| Random seeds | 10, 20, 30 (fixed for reproducibility — MOD-3) |
| Checkpoint A | Best validation AUROC (fold 9) — `best_resnet_{exp}.pth` |
| Checkpoint B | Final epoch (epoch 50) — `final_resnet_{exp}.pth` |
| Bootstrap CI | 100 iterations, 95% CI (diagnostic estimate) |

**Modifications from original source (MODs):**

- **MOD-1**: 12-lead enforcement — signals with fewer than 12 leads are zero-padded; signals with more than 12 leads are truncated. Not present in original source.
- **MOD-2**: Per-record error handling — individual records that fail to load are skipped. Not present in original source.
- **MOD-3**: Fixed random seeds (10, 20, 30) — Strodthoff et al. (2021) do not specify a fixed random seed. Seeds were added for the M2 three-seed stability analysis.

---

## M2 Closure Results

### Item A — Checkpoint Rule Reconcile

Both checkpoint rules were evaluated side by side for all three seeds. The supervisor is asked to confirm which rule constitutes the M2 primary criterion.

| Run | Seed | exp0 Ckpt A (best-val) | exp0 Ckpt B (final-ep) | exp3 Ckpt A (best-val) | exp3 Ckpt B (final-ep) |
|-----|------|----------------------|----------------------|----------------------|----------------------|
| 1 | 10 | 0.9097 | 0.9107 | 0.9340 | 0.9337 |
| 2 | 20 | 0.9172 | 0.9142 | 0.9324 | 0.9319 |
| 3 | 30 | 0.9204 | 0.9191 | 0.9265 | 0.9292 |
| **Mean±SD** | | **0.9158±0.0055** | **0.9147±0.0042** | **0.9310±0.0040** | **0.9316±0.0023** |
| Target | | 0.9190 | 0.9190 | 0.9460 | 0.9460 |
| Diff | | −0.35% | −0.47% | −1.59% | −1.52% |

The checkpoint difference is small relative to the observed variation across the three seeds, and no consistent directional advantage is observed for either checkpoint rule.

### Item B — Three-Seed Stability

Three independent runs of the exact V7 configuration using fixed seeds 10, 20, 30. Results reported as mean ± SD (ddof=1). Never best-of-N.

- **exp0**: mean = 0.9158 ± 0.0055 (Ckpt A) | diff = −0.35% — **PASS**
- **exp3**: mean = 0.9310 ± 0.0040 (Ckpt A) | diff = −1.59% — **DOCUMENTED** (within metric resolution per supervisor email)

### Item C — Per-Class AUROC exp3

Per-class AUROC for exp3 on fold 10, Checkpoint A, seed=30.

| Class | AUROC | Positives (fold 10) | Note |
|-------|-------|-------------------|------|
| AFIB | 0.9861 | 152 | |
| AFLT | 0.9672 | 7 | < 20 positives |
| BIGU | 0.7952 | 8 | < 20 positives |
| PACE | 0.9523 | 29 | |
| PSVT | 0.9995 | 2 | < 20 positives |
| SARRH | 0.7265 | 77 | |
| SBRAD | 0.9362 | 64 | |
| SR | 0.8754 | 1678 | |
| STACH | 0.9918 | 82 | |
| SVARR | 0.9508 | 14 | < 20 positives |
| SVTAC | 0.9790 | 3 | < 20 positives |
| TRIGU | 0.9581 | 2 | < 20 positives |
| **Macro** | **0.9265** | | |

The lower AUROC observed for SARRH and SR indicates that the residual is not confined to classes with very small positive counts.

---

## Reproducibility Notes

The original benchmark was executed using an older software environment, including FastAI v1 and an older PyTorch version.

The current work reproduces the documented behaviour in a modern PyTorch environment. Therefore, exact numerical identity is not assumed where framework-level implementation details, random number generation, optimisation behaviour, or historical library versions may differ.

The development history has been retained to distinguish:

- benchmark reproduction,
- diagnostic investigation,
- exploratory experiments, and
- source-supported corrections.

---

## Environment

| Package | Version |
|---------|---------|
| Python | 3.11.9 |
| torch | 2.5.1+cu121 |
| numpy | 1.26.4 |
| scikit-learn | 1.8.0 |
| wfdb | 4.3.1 |
| pandas | 2.2.3 |
| scipy | 1.17.1 |
| tqdm | 4.67.3 |
| wandb | 0.27.0 |
| CUDA | 12.1 |
| GPU | NVIDIA GeForce RTX 4060 Laptop (8 GB VRAM) |
| OS | Windows 11 |

See `requirements (1).txt` for pinned package versions from the experiment environment.

---

## License

This repository is published under the GNU General Public License v3.0 (GPL-3.0), consistent with the original source repository (github.com/helme/ecg_ptbxl_benchmarking, GPL-3.0, Copyright (C) 2020 Patrick Wagner).

---

## Repository Contents

The repository contains only the scripts and result summaries required to document the reproduction.

Large ECG waveform files, cached arrays, trained model checkpoints, and the Python virtual environment are excluded.

---

## Author

Yaser Karimi
PhD in Computing
University of Staffordshire
