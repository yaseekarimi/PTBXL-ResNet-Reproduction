# PTB-XL ResNet1d_Wang Benchmark Reproduction

## Phase 0 – M2

This repository documents the reproduction and diagnostic analysis of the
`resnet1d_wang` benchmark reported by Strodthoff et al. (2021) on the PTB-XL
dataset.

The objective was to reproduce the published benchmark under comparable
experimental conditions before introducing modifications for later stages of
the PhD research.

The two benchmark tasks investigated were:

- **exp0 (all):** multi-label classification of 71 SCP statements
- **exp3 (rhythm):** multi-label classification of 12 rhythm statements

Published macro-AUROC reference values:

| Experiment | Task | Published AUROC |
|---|---|---:|
| exp0 | All SCP statements (71 classes) | 0.919 |
| exp3 | Rhythm statements (12 classes) | 0.946 |

---

## Reference Implementation

The reproduction was based on the paper and official source repository:

**Strodthoff, N., Wagner, P., Schaeffter, T. and Samek, W. (2021).**
*Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL.*
IEEE Journal of Biomedical and Health Informatics, 25(5), 1519–1528.

DOI: 10.1109/JBHI.2020.3022989

Official implementation:
https://github.com/helme/ecg_ptbxl_benchmarking

The following source files from the official implementation were particularly
important during the source audit:

- `fastai_configs.py`
- `fastai_model.py`
- `basic_conv1d.py`
- `timeseries_utils.py`
- `scp_experiment.py`
- `utils.py`
- `ecg_env.yml`

The original implementation used FastAI v1. The reproduction in this
repository was implemented in modern PyTorch while progressively auditing the
implementation against the original source code.

---

## Dataset

Dataset: **PTB-XL**

Reference:

Wagner, P. et al. (2020).
*PTB-XL, a large publicly available electrocardiography dataset.*
Scientific Data, 7, 154.

DOI: 10.1038/s41597-020-0495-6

For the final source-audited reproduction:

- PTB-XL version: **v1.0.1**
- Records: **21,837**
- Sampling rate: **100 Hz**
- Signal source: `filename_lr` / `records100/`
- ECG duration: **10 seconds**
- Leads: **12**

PTB-XL data are not included in this repository.

---

## Why PTB-XL v1.0.1 Was Downloaded Again

Earlier experiments were performed using PTB-XL v1.0.3, which contained
21,799 available records in the local experimental pipeline.

During investigation of the remaining reproduction gap, the dataset version
was audited against the benchmark setup. PTB-XL v1.0.1 contained 21,837
records, resulting in a difference of 38 records.

The missing records were distributed across the official folds, including
validation fold 9 and test fold 10. Differences were also observed in some
rhythm-label counts.

Therefore, the complete 100-Hz PTB-XL v1.0.1 dataset was downloaded and the
experiments were repeated without changing the model.

This step was performed to eliminate dataset-version mismatch as a possible
source of the difference from the published benchmark.

---

## Data Split

The official PTB-XL stratified fold protocol was used:

- **Folds 1–8:** Training
- **Fold 9:** Validation
- **Fold 10:** Test

For PTB-XL v1.0.1:

### exp0

- Training: 17,441
- Validation: 2,193
- Test: 2,203

### exp3

- Training: 16,854
- Validation: 2,109
- Test: 2,103

---

## Signal Preprocessing

The reproduction uses the 100-Hz PTB-XL signals.

No additional high-pass, notch, or band-pass filtering was introduced because
the benchmark-specific preprocessing in the reference implementation uses
standardisation rather than additional ECG filtering.

The preprocessing pipeline was separately verified before the benchmark
experiments.

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

For exp0, three of the five runs were within the defined 0.01 reproduction
criterion. The mean test AUROC was approximately 0.9109.

For exp3, a larger and persistent difference from the published benchmark was
observed.

### Exploratory learning-rate experiment

An exploratory run using `max_lr=0.003` was performed to investigate whether
the learning-rate schedule explained the exp3 discrepancy.

The experiment did not resolve the exp3 gap and was treated as a diagnostic
experiment rather than the final reproduction.

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

These source-supported corrections were incorporated into the subsequent
implementation.

### v2–v5 diagnostic stages

Further experiments investigated:

- 50 versus 100 training epochs
- FastAI-style OneCycle behaviour
- beta1/momentum scheduling
- per-class exp3 AUROC
- PTB-XL dataset-version differences
- PTB-XL v1.0.1 reproduction

These experiments progressively reduced uncertainty about the remaining
difference from the published benchmark.

### v6 → v7 source audit

The final v6-to-v7 transition was based on a direct audit of the original
source implementation.

The main source-supported corrections were:

1. Exact weight initialisation behaviour, including bias and BatchNorm
   initialisation.
2. Training crop generation changed to reproduce the Python
   `random.randint()` behaviour used in `TimeseriesDatasetCrops`.
3. Final-epoch evaluation treated as the primary source-faithful result,
   because the default official configuration saves the learner after
   `fit_one_cycle()` and does not introduce the additional validation-based
   early-stopping rule used in earlier diagnostic versions.

No new architecture, loss function, crop size, learning rate, epoch count, or
other performance-oriented modification was introduced in v7.

---

## Final v7 Configuration

| Parameter | Final configuration |
|---|---|
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
| LR schedule | One-cycle |
| Filtering | None |
| Train folds | 1–8 |
| Validation fold | 9 |
| Test fold | 10 |
| Evaluation | Macro-AUROC |

---

## Final v7 Results

| Experiment | Published | v7 Test AUROC | Difference | 95% Bootstrap CI |
|---|---:|---:|---:|---:|
| exp0 (all) | 0.919 | **0.9153** | -0.0037 | [0.9063, 0.9235] |
| exp3 (rhythm) | 0.946 | **0.9311** | -0.0149 | [0.9170, 0.9471] |

### Interpretation

For **exp0**, the final source-audited reproduction produced a macro-AUROC of
0.9153 compared with the published value of 0.919.

For **exp3**, the final result was 0.9311 compared with the published value of
0.946. The remaining discrepancy was not addressed through arbitrary
hyperparameter tuning against fold 10.

The v7 experiment therefore represents the current source-audited PyTorch
reimplementation and provides a documented basis for discussing the remaining
difference from the historical FastAI-v1 implementation.

---

## Reproducibility Notes

The original benchmark was executed using an older software environment,
including FastAI v1 and an older PyTorch version.

The current work reproduces the documented behaviour in a modern PyTorch
environment. Therefore, exact numerical identity is not assumed where
framework-level implementation details, random number generation, optimisation
behaviour, or historical library versions may differ.

The development history has been retained to distinguish:

- benchmark reproduction,
- diagnostic investigation,
- exploratory experiments, and
- source-supported corrections.

---

## Repository Contents

The repository contains only the scripts and result summaries required to
document the reproduction.

Large ECG waveform files, cached arrays, trained model checkpoints, and the
Python virtual environment are excluded.

---

## Author

Yaser Karimi  
PhD in Computing  
University of Staffordshire
