"""
ResNet1d_wang — exp0 (all) and exp3 (rhythm) Reproduction
======================================================
Phase 0 - Month 2

Purpose:
    Reproduces the exp0 (all statements) and exp3 (rhythm) benchmark
    tasks reported by Strodthoff et al. (2021) on PTB-XL using the
    resnet1d_wang architecture, 100 Hz sampling rate, official fold
    protocol, and multi-label classification with Bootstrap CI.

    Target:
        exp0 (all)    : macro AUROC within 1% of published result
        exp3 (rhythm) : macro AUROC within 1% of published result

    Published results (Strodthoff et al., 2021, Table II):
        resnet1d_wang exp0: ~0.919
        resnet1d_wang exp3: ~0.946

References:
-----------
[1] Strodthoff, N., Wagner, P., Schaeffter, T. and Samek, W. (2021)
    'Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL',
    IEEE Journal of Biomedical and Health Informatics, 25(5), pp. 1519-1528.
    DOI: 10.1109/JBHI.2020.3022989
    GitHub: https://github.com/helme/ecg_ptbxl_benchmarking
    Original files:
        code/utils/utils.py
        code/models/resnet1d.py
        code/experiments/scp_experiment.py

[2] Wagner, P. et al. (2020)
    'PTB-XL, a large publicly available electrocardiography dataset',
    Scientific Data, 7(1), p. 154.
    DOI: 10.1038/s41597-020-0495-6

[3] He, K., Zhang, X., Ren, S. and Sun, J. (2016)
    'Deep Residual Learning for Image Recognition',
    Proceedings of CVPR, pp. 770-778.
    DOI: 10.1109/CVPR.2016.90

[4] Smith, L.N. and Topin, N. (2019)
    'Super-Convergence: Very Fast Training of Neural Networks
    Using Large Learning Rates',
    Proceedings of SPIE, 11006.
    DOI: 10.1117/12.2520589

[5] Kingma, D.P. and Ba, J. (2015)
    'Adam: A Method for Stochastic Optimization',
    ICLR. arXiv: 1412.6980

[6] Efron, B. and Tibshirani, R.J. (1993)
    'An Introduction to the Bootstrap',
    Chapman and Hall/CRC.
    ISBN: 978-0412042317

[7] Goldberger, A.L. et al. (2000)
    'PhysioBank, PhysioToolkit, and PhysioNet',
    Circulation, 101(23), pp. e215-e220.
    DOI: 10.1161/01.CIR.101.23.e215

[8] Loshchilov, I. and Hutter, F. (2019)
    'Decoupled Weight Decay Regularization',
    ICLR. arXiv: 1711.05101

Adaptation notes:
-----------------
- load_raw_data_ptbxl() adapted from Strodthoff et al. (2021),
  utils.py: loads PTB-XL at 100 Hz via filename_lr.

- compute_label_aggregations() adapted from Strodthoff et al. (2021),
  utils.py: extracts SCP statement labels for exp0 (all) and
  exp3 (rhythm) using scp_statements.csv.

- select_data() adapted from Strodthoff et al. (2021),
  utils.py: converts labels to multi-hot encoding via
  MultiLabelBinarizer.

- preprocess_signals() adapted from Strodthoff et al. (2021),
  utils.py: fits StandardScaler on training data and applies
  to all splits.

- apply_standardizer() adapted from Strodthoff et al. (2021),
  utils.py: applies StandardScaler per record.

- get_appropriate_bootstrap_samples() adapted from Strodthoff
  et al. (2021), utils.py: generates bootstrap samples ensuring
  all classes represented [Efron and Tibshirani, 1993].

- evaluate_experiment() adapted from Strodthoff et al. (2021),
  utils.py: computes macro AUROC via roc_auc_score.

- BasicBlock1d and ResNet1d adapted from Strodthoff et al. (2021),
  code/models/resnet1d.py, resnet1d_wang variant — confirmed.
  Residual connection design follows He et al. (2016).

- Fold protocol adapted from Strodthoff et al. (2021),
  code/experiments/scp_experiment.py:
  folds 1-8 train, fold 9 validation, fold 10 test.

- One-cycle LR scheduler follows Smith and Topin (2019).
- Adam optimiser follows Kingma and Ba (2015).

Modifications from original Strodthoff et al. (2021) source:
--------------------------------------------------------------
MOD-1: 12-lead enforcement added. Signals with fewer than 12
        leads are zero-padded; signals with more than 12 leads
        are truncated. Original source assumes all PTB-XL
        recordings have exactly 12 leads.

MOD-2: Per-record error handling added. If an individual record
        fails to load, it is skipped and processing continues.
        Original source uses vectorised loading without explicit
        per-record error handling.

MOD-3: Classification head updated to match original source.
        AdaptiveConcatPool1d (Max+Avg concatenation) replaces
        AdaptiveAvgPool1d. Hidden layer 256→128 and ps_head=0.5
        dropout added. Adapted from basic_conv1d.py, create_head1d().

MOD-4: Learning rate updated to 0.01 — confirmed default in
        fastai_model.py, consistent with parameters: dict() in
        the benchmark configuration (no override).

MOD-5: Source weight initialisation reproduced from basic_conv1d.py:
        Kaiming normal for Conv1d/Linear, zero bias when present, and
        BatchNorm1d weight=1/bias=0. Reference: official repository.

MOD-6: Crop-based prediction added. Original uses
        TimeseriesDatasetCrops with input_size=2.5s (250 samples),
        random crops for training and overlapping crops (stride=125)
        for validation/test, with max aggregation across crops.
        Adapted from timeseries_utils.py.

MOD-7: FastAI-v1 true weight decay retained: wd=0.01 is applied
        manually as decoupled decay using the current batch LR. This
        avoids substituting modern AdamW semantics for the historical
        FastAI-v1 Learner behaviour.
"""

import os
import ast
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import wfdb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ============================================================
# CONFIG
# ============================================================
PTBXL_PATH = "data/ptbxl/ptb-xl-1.0.1/"
OUTPUT_DIR  = "results/exp_reproduction/"

SAMPLING_RATE = 100   # Hz — Strodthoff et al. (2021)
TARGET_LEN    = 1000  # 10s × 100Hz
N_LEADS       = 12

BATCH_SIZE = 128      # Strodthoff et al. (2021)
EPOCHS     = 50    # Confirmed default in fastai_model.py [Strodthoff et al., 2021]
MAX_LR     = 0.01     # Confirmed default in fastai_model.py [Strodthoff et al., 2021]
WEIGHT_DECAY = 0.01   # Confirmed default in fastai_model.py [Strodthoff et al., 2021]
CROP_LEN   = 250      # 2.5s × 100Hz — TimeseriesDatasetCrops [Strodthoff et al., 2021]
CROP_STRIDE = 125     # Overlapping crops for val/test [Strodthoff et al., 2021]
N_BOOTSTRAP = 100     # Temporary: 10,000 for final run only [Strodthoff et al., 2021]

EXPERIMENTS = {
    'exp0': 'all',      # all SCP statements
    'exp3': 'rhythm',   # rhythm statements only
}

# Published targets [Strodthoff et al., 2021, Table II]
TARGETS = {
    'exp0': 0.919,
    'exp3': 0.946,
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print("ResNet1d_wang — exp0 (all) + exp3 (rhythm)")
print("Adapted from Strodthoff et al. (2021)")
print("100 Hz | No filter | Fold 1-8/9/10 | Multi-label | BCE")
print("v7: source-audited FastAI-v1 reproduction | exact weight_init + exact crop RNG | final-epoch primary")
print("=" * 65)


# ============================================================
# DATA LOADING
# Adapted from Strodthoff et al. (2021), utils.py:
#   load_raw_data_ptbxl()
# ============================================================
def load_raw_data_ptbxl(df, sampling_rate, path):
    """
    Load PTB-XL ECG signals at specified sampling rate.

    Adapted from Strodthoff et al. (2021), utils.py,
    load_raw_data_ptbxl():
        data = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_lr)]

    MOD-1: 12-lead enforcement added per record.
    MOD-2: Per-record error handling added.

    Parameters
    ----------
    df : pd.DataFrame — PTB-XL metadata
    sampling_rate : int — 100 or 500
    path : str — PTB-XL root directory

    Returns
    -------
    np.ndarray, shape (n_records, n_samples, 12)
    """
    print(f"\nLoading PTB-XL at {sampling_rate} Hz...")

    # Check for cached version [Strodthoff et al., 2021]
    cache_path = path + f'raw{sampling_rate}.npy'
    if os.path.exists(cache_path):
        print(f"  Loading from cache: {cache_path}")
        return np.load(cache_path, allow_pickle=True)

    data = []
    errors = 0

    # filename_lr for 100 Hz [Strodthoff et al., 2021]
    filenames = df.filename_lr if sampling_rate == 100 else df.filename_hr

    for fname in tqdm(filenames, desc="Loading"):
        try:
            # Adapted from load_raw_data_ptbxl() [Strodthoff et al., 2021]
            record = wfdb.rdsamp(path + fname)
            sig = record[0]  # (n_samples, n_leads)

            # MOD-1: 12-lead enforcement
            if sig.shape[1] < N_LEADS:
                pad = np.zeros((sig.shape[0], N_LEADS - sig.shape[1]))
                sig = np.concatenate([sig, pad], axis=1)
            elif sig.shape[1] > N_LEADS:
                sig = sig[:, :N_LEADS]

            data.append(sig)

        except Exception:
            # MOD-2: per-record error handling
            data.append(np.zeros((TARGET_LEN, N_LEADS)))
            errors += 1

    data = np.array(data)
    print(f"  Loaded: {len(data)} records | Errors: {errors} [MOD-2]")

    # Cache for future runs [Strodthoff et al., 2021]
    data.dump(cache_path)
    return data


# ============================================================
# LABEL AGGREGATION
# Adapted from Strodthoff et al. (2021), utils.py:
#   compute_label_aggregations()
# ============================================================
def compute_label_aggregations(df, folder, ctype):
    """
    Aggregate SCP codes into task-specific label sets.

    Adapted from Strodthoff et al. (2021), utils.py,
    compute_label_aggregations():
        aggregation_df = pd.read_csv(folder+'scp_statements.csv')

    For exp0 (all): all SCP statement keys retained.
    For exp3 (rhythm): only SCP codes with rhythm==1.0 retained.

    Parameters
    ----------
    df : pd.DataFrame
    folder : str — path to scp_statements.csv
    ctype : str — 'all' or 'rhythm'

    Returns
    -------
    pd.DataFrame with aggregated labels column
    """
    # Load SCP statement metadata [Strodthoff et al., 2021]
    aggregation_df = pd.read_csv(folder + 'scp_statements.csv', index_col=0)

    if ctype == 'all':
        # exp0: all SCP statement keys [Strodthoff et al., 2021]
        df['all_scp'] = df.scp_codes.apply(lambda x: list(set(x.keys())))

    elif ctype == 'rhythm':
        # exp3: rhythm statements only [Strodthoff et al., 2021]
        rhythm_agg_df = aggregation_df[aggregation_df.rhythm == 1.0]

        def aggregate_rhythm(y_dic):
            # Adapted from Strodthoff et al. (2021), utils.py
            tmp = []
            for key in y_dic.keys():
                if key in rhythm_agg_df.index:
                    c = key
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        df['rhythm'] = df.scp_codes.apply(aggregate_rhythm)
        df['rhythm_len'] = df.rhythm.apply(lambda x: len(x))

    return df


# ============================================================
# DATA SELECTION + MULTI-HOT ENCODING
# Adapted from Strodthoff et al. (2021), utils.py:
#   select_data()
# ============================================================
def select_data(XX, YY, ctype, min_samples=0):
    """
    Select records with valid labels and convert to multi-hot.

    Adapted from Strodthoff et al. (2021), utils.py,
    select_data(): uses MultiLabelBinarizer for multi-hot encoding.

    Parameters
    ----------
    XX : np.ndarray — signals
    YY : pd.DataFrame — labels
    ctype : str — 'all' or 'rhythm'
    min_samples : int — minimum samples per class

    Returns
    -------
    tuple : (X, Y, y_multihot, mlb)
    """
    mlb = MultiLabelBinarizer()

    if ctype == 'all':
        # exp0 [Strodthoff et al., 2021]
        counts = pd.Series(
            np.concatenate(YY.all_scp.values)
        ).value_counts()
        counts = counts[counts > min_samples]
        YY = YY.copy()
        YY.all_scp = YY.all_scp.apply(
            lambda x: list(set(x).intersection(set(counts.index.values)))
        )
        YY['all_scp_len'] = YY.all_scp.apply(lambda x: len(x))
        X = XX[YY.all_scp_len > 0]
        Y = YY[YY.all_scp_len > 0]
        mlb.fit(Y.all_scp.values)
        y = mlb.transform(Y.all_scp.values)

    elif ctype == 'rhythm':
        # exp3 [Strodthoff et al., 2021]
        counts = pd.Series(
            np.concatenate(YY.rhythm.values)
        ).value_counts()
        counts = counts[counts > min_samples]
        YY = YY.copy()
        YY.rhythm = YY.rhythm.apply(
            lambda x: list(set(x).intersection(set(counts.index.values)))
        )
        YY['rhythm_len'] = YY.rhythm.apply(lambda x: len(x))
        X = XX[YY.rhythm_len > 0]
        Y = YY[YY.rhythm_len > 0]
        mlb.fit(Y.rhythm.values)
        y = mlb.transform(Y.rhythm.values)

    print(f"  Classes: {len(mlb.classes_)} | Records: {len(X)}")
    print(f"  Classes: {list(mlb.classes_)}")
    return X, Y, y, mlb


# ============================================================
# SIGNAL NORMALISATION
# Adapted from Strodthoff et al. (2021), utils.py:
#   preprocess_signals() + apply_standardizer()
# ============================================================
def apply_standardizer(X, ss):
    """
    Apply StandardScaler to all records.

    Adapted from Strodthoff et al. (2021), utils.py,
    apply_standardizer():
        X_tmp.append(ss.transform(x.flatten()[:,np.newaxis]).reshape(x_shape))

    Parameters
    ----------
    X : np.ndarray, shape (n, n_samples, 12)
    ss : fitted StandardScaler

    Returns
    -------
    np.ndarray, normalised
    """
    X_tmp = []
    for x in X:
        x_shape = x.shape
        # Adapted from Strodthoff et al. (2021), utils.py
        X_tmp.append(
            ss.transform(x.flatten()[:, np.newaxis]).reshape(x_shape)
        )
    return np.array(X_tmp)


def preprocess_signals(X_train, X_val, X_test):
    """
    Fit StandardScaler on training data and apply to all splits.

    Adapted from Strodthoff et al. (2021), utils.py,
    preprocess_signals():
        ss = StandardScaler()
        ss.fit(np.vstack(X_train).flatten()[:,np.newaxis].astype(float))

    Parameters
    ----------
    X_train, X_val, X_test : np.ndarray

    Returns
    -------
    tuple : (X_train_norm, X_val_norm, X_test_norm, ss)
    """
    # Fit on training data only [Strodthoff et al., 2021]
    ss = StandardScaler()
    ss.fit(np.vstack(X_train).flatten()[:, np.newaxis].astype(float))
    return (
        apply_standardizer(X_train, ss),
        apply_standardizer(X_val,   ss),
        apply_standardizer(X_test,  ss),
        ss
    )


# ============================================================
# BOOTSTRAP CI
# Adapted from Strodthoff et al. (2021), utils.py:
#   get_appropriate_bootstrap_samples()
# Source: Efron and Tibshirani (1993)
# ============================================================
def get_appropriate_bootstrap_samples(y_true, n_bootstrapping_samples):
    """
    Generate bootstrap sample indices ensuring all classes represented.

    Adapted from Strodthoff et al. (2021), utils.py,
    get_appropriate_bootstrap_samples().
    Bootstrap resampling method: Efron and Tibshirani (1993).

    Parameters
    ----------
    y_true : np.ndarray, shape (n, n_classes)
    n_bootstrapping_samples : int

    Returns
    -------
    list of np.ndarray — bootstrap sample indices
    """
    samples = []
    while True:
        # Random sampling with replacement [Efron & Tibshirani, 1993]
        ridxs = np.random.randint(0, len(y_true), len(y_true))
        if y_true[ridxs].sum(axis=0).min() != 0:
            samples.append(ridxs)
            if len(samples) == n_bootstrapping_samples:
                break
    return samples


# ============================================================
# EVALUATION
# Adapted from Strodthoff et al. (2021), utils.py:
#   evaluate_experiment() + generate_results()
# ============================================================
def evaluate_experiment(y_true, y_pred):
    """
    Compute macro AUROC.

    Adapted from Strodthoff et al. (2021), utils.py,
    evaluate_experiment():
        results['macro_auc'] = roc_auc_score(y_true, y_pred,
                                              average='macro')

    Parameters
    ----------
    y_true : np.ndarray, shape (n, n_classes)
    y_pred : np.ndarray, shape (n, n_classes)

    Returns
    -------
    float : macro AUROC
    """
    # Adapted from Strodthoff et al. (2021), utils.py
    return roc_auc_score(y_true, y_pred, average='macro')


def compute_bootstrap_ci(y_true, y_pred, n_samples=N_BOOTSTRAP):
    """
    Compute 95% Bootstrap CI for macro AUROC.

    Adapted from Strodthoff et al. (2021), utils.py,
    evaluate() method in scp_experiment.py.
    Bootstrap method: Efron and Tibshirani (1993).

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray
    n_samples : int

    Returns
    -------
    tuple : (mean, lower_95, upper_95)
    """
    # Point estimate
    point = evaluate_experiment(y_true, y_pred)

    # Bootstrap samples [Efron & Tibshirani, 1993]
    boot_samples = get_appropriate_bootstrap_samples(y_true, n_samples)
    boot_scores  = [
        evaluate_experiment(y_true[idx], y_pred[idx])
        for idx in boot_samples
    ]

    mean  = np.mean(boot_scores)
    lower = np.percentile(boot_scores, 2.5)
    upper = np.percentile(boot_scores, 97.5)

    return point, mean, lower, upper


# ============================================================
# PYTORCH DATASET — with crop support
# Adapted from TimeseriesDatasetCrops [Strodthoff et al., 2021]
# timeseries_utils.py
# ============================================================
class ECGDataset(Dataset):
    """
    ECG Dataset with random crop support for training.
    Adapted from TimeseriesDatasetCrops [Strodthoff et al., 2021],
    timeseries_utils.py.

    MOD-6: crop_len and random_crop added for training.
    For val/test full signal used (crops handled separately).
    """
    def __init__(self, X, y, crop_len=None, random_crop=False):
        # Transpose to (n, 12, n_samples) for Conv1d
        self.X = torch.tensor(
            X.transpose(0, 2, 1), dtype=torch.float32
        )
        self.y = torch.tensor(y, dtype=torch.float32)
        self.crop_len    = crop_len
        self.random_crop = random_crop

    def __len__(self): return len(self.X)

    def __getitem__(self, i):
        x = self.X[i]  # (12, n_samples)
        if self.crop_len is not None and self.random_crop:
            # Random crop for training [Strodthoff et al., 2021]
            # Adapted from TimeseriesDatasetCrops.__getitem__()
            # Official TimeseriesDatasetCrops uses Python random.randint
            # with upper bound (signal_len - crop_len - 1), inclusive.
            # For 1000 samples and crop_len=250 this gives starts 0..749.
            max_start = x.shape[1] - self.crop_len - 1
            start = random.randint(0, max_start) if max_start >= 0 else 0
            x = x[:, start:start + self.crop_len]
        return x, self.y[i]


def get_crop_predictions(model, X, y, crop_len, stride, batch_size, device):
    """
    Generate predictions using overlapping crops with max aggregation.

    Adapted from TimeseriesDatasetCrops and aggregate_predictions()
    in timeseries_utils.py [Strodthoff et al., 2021].

    For each record: extract overlapping crops of crop_len with given
    stride, predict each crop, aggregate by taking max across crops.
    MOD-6: replaces single-prediction approach.

    Parameters
    ----------
    model     : trained ResNet1d
    X         : np.ndarray, shape (n, n_samples, 12)
    y         : np.ndarray, shape (n, n_classes)
    crop_len  : int — crop length in samples (250 for 2.5s at 100Hz)
    stride    : int — stride between crops (125 for 50% overlap)
    batch_size: int
    device    : torch.device

    Returns
    -------
    np.ndarray, shape (n, n_classes) — max-aggregated predictions
    """
    model.eval()
    n_records = len(X)
    n_classes = y.shape[1]
    all_preds = np.zeros((n_records, n_classes), dtype=np.float32)

    with torch.no_grad():
        for i in range(n_records):
            sig = X[i]  # (n_samples, 12)
            n_samples = sig.shape[0]

            # Generate overlapping crops [Strodthoff et al., 2021]
            # Adapted from TimeseriesDatasetCrops (chunkify_valid=True)
            starts = list(range(0, n_samples - crop_len + 1, stride))
            if not starts:
                starts = [0]

            crops = []
            for s in starts:
                crop = sig[s:s + crop_len, :]  # (crop_len, 12)
                crop_t = torch.tensor(
                    crop.T, dtype=torch.float32
                ).unsqueeze(0)  # (1, 12, crop_len)
                crops.append(crop_t)

            # Batch predict all crops
            crops_tensor = torch.cat(crops, dim=0).to(device)
            preds = torch.sigmoid(model(crops_tensor))  # (n_crops, n_classes)

            # Max aggregation across crops [Strodthoff et al., 2021]
            # Adapted from aggregate_predictions(), aggregate_fn='max'
            all_preds[i] = preds.max(dim=0).values.cpu().numpy()

    return all_preds


# ============================================================
# RESNET1D ARCHITECTURE
# Adapted from Strodthoff et al. (2021)
# GitHub: code/models/resnet1d.py — resnet1d_wang variant
# Residual connection: He et al. (2016)
# ============================================================
def conv1d(in_planes, out_planes, stride=1, kernel_size=3):
    """
    1D convolution with padding.
    Adapted from Strodthoff et al. (2021), resnet1d.py.
    """
    return nn.Conv1d(
        in_planes, out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=(kernel_size - 1) // 2,
        bias=False
    )


class BasicBlock1d(nn.Module):
    """
    1D Basic Residual Block.
    Adapted from Strodthoff et al. (2021), resnet1d.py.
    Residual connection: He et al. (2016).
    MOD-1: No changes to block architecture.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,
                 kernel_size=3, downsample=None):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size // 2 + 1]

        self.conv1      = conv1d(inplanes, planes, stride=stride,
                                 kernel_size=kernel_size[0])
        self.bn1        = nn.BatchNorm1d(planes)
        self.relu       = nn.ReLU(inplace=True)
        self.conv2      = conv1d(planes, planes,
                                 kernel_size=kernel_size[1])
        self.bn2        = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride     = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # Residual connection [He et al., 2016]
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class AdaptiveConcatPool1d(nn.Module):
    """
    Concatenation of AdaptiveMaxPool1d and AdaptiveAvgPool1d.

    Adapted from basic_conv1d.py, AdaptiveConcatPool1d()
    [Strodthoff et al., 2021]:
        self.ap = nn.AdaptiveAvgPool1d(sz)
        self.mp = nn.AdaptiveMaxPool1d(sz)
        return torch.cat([self.mp(x), self.ap(x)], 1)

    MOD-3: replaces AdaptiveAvgPool1d in classification head.
    Output features: 2 * inplanes (Max + Avg concatenated).
    """
    def __init__(self, sz=1):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool1d(sz)
        self.mp = nn.AdaptiveMaxPool1d(sz)

    def forward(self, x):
        # Adapted from [Strodthoff et al., 2021], basic_conv1d.py
        return torch.cat([self.mp(x), self.ap(x)], 1)


class ResNet1d(nn.Module):
    """
    ResNet1d — resnet1d_wang variant — confirmed.
    Adapted from Strodthoff et al. (2021), resnet1d.py.

    Architecture:
        Stem   : Conv1d, kernel=7, stride=1, 128 channels
        Block 1: 128ch, kernel=[5,3]
        Block 2: 128ch, kernel=[5,3]
        Block 3: 128ch, kernel=[5,3]
        Head   : AdaptiveAvgPool1d → Flatten → Linear(n_classes)

    MOD-1 (fastai→PyTorch): AdaptiveConcatPool1d replaced with
        AdaptiveAvgPool1d + Flatten + Linear.
    """
    def __init__(self, input_channels=12, num_classes=71,
                 inplanes=128, kernel_size=None,
                 kernel_size_stem=7, stride_stem=1):
        super().__init__()
        self.inplanes = inplanes
        if kernel_size is None:
            kernel_size = [5, 3]  # resnet1d_wang [Strodthoff et al., 2021]

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, inplanes,
                      kernel_size=kernel_size_stem,
                      stride=stride_stem,
                      padding=(kernel_size_stem - 1) // 2,
                      bias=False),
            nn.BatchNorm1d(inplanes),
            nn.ReLU(inplace=True)
        )

        # 3 residual blocks [Strodthoff et al., 2021]
        self.layer1 = self._make_layer(BasicBlock1d, inplanes, 1,
                                       kernel_size=kernel_size)
        self.layer2 = self._make_layer(BasicBlock1d, inplanes, 1,
                                       kernel_size=kernel_size)
        self.layer3 = self._make_layer(BasicBlock1d, inplanes, 1,
                                       kernel_size=kernel_size)

        # Head — MOD-3: AdaptiveConcatPool1d + hidden layer + dropout
        # Adapted from create_head1d() in basic_conv1d.py
        # [Strodthoff et al., 2021]
        # concat_pooling=True: Max + Avg → 2*inplanes features
        # lin_ftrs_head=[128]: hidden layer 256→128
        # ps_head=0.5: dropout probability
        nf = 2 * inplanes  # 256 — ConcatPool doubles features
        self.pool = AdaptiveConcatPool1d()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(nf),
            nn.Dropout(p=0.25),        # fastai ps_head=0.5 → first layer gets ps/2
            nn.Linear(nf, 128),        # 256 → 128
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.5),         # fastai ps_head=0.5 → last layer gets ps
            nn.Linear(128, num_classes)
            # No sigmoid — BCEWithLogitsLoss handles it
        )

    def _make_layer(self, block, planes, blocks,
                    stride=1, kernel_size=3):
        """Adapted from ResNet1d._make_layer [Strodthoff et al., 2021]."""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride,
                        kernel_size, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                kernel_size=kernel_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)   # AdaptiveConcatPool1d — MOD-3
        return self.head(x)


# ============================================================
# WEIGHT INITIALISATION
# Adapted from weight_init() in fastai_model.py
# [Strodthoff et al., 2021]
# MOD-5: custom Kaiming initialisation
# ============================================================
def weight_init(m):
    """Source-faithful weight_init from official basic_conv1d.py.

    Reference:
      helme/ecg_ptbxl_benchmarking/code/models/basic_conv1d.py

    Official behaviour:
      - Conv1d / Linear: Kaiming-normal weights; zero bias if present.
      - BatchNorm1d: weight=1; bias=0.
    """
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class FastAIV1OneCycle:
    """Minimal source-faithful FastAI-v1 OneCycle schedule.

    Mirrors fastai1 OneCycleScheduler/Scheduler semantics:
      n = len(train_dl) * epochs
      phase1 = int(n * pct_start)
      phase2 = n - phase1
      LR: max_lr/div_factor -> max_lr -> max_lr/(div_factor*1e4)
      beta1: 0.95 -> 0.85 -> 0.95
      cosine annealing; schedule advances at the END of each train batch.
    """
    def __init__(self, optimizer, total_steps, max_lr, pct_start=0.3,
                 div_factor=25.0, moms=(0.95, 0.85)):
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.phase1 = int(self.total_steps * pct_start)
        self.phase2 = self.total_steps - self.phase1
        self.max_lr = float(max_lr)
        self.low_lr = self.max_lr / float(div_factor)
        self.final_lr = self.max_lr / (float(div_factor) * 1e4)
        self.mom_hi, self.mom_lo = float(moms[0]), float(moms[1])
        self.phase = 0
        self.n = 0
        self._set(self.low_lr, self.mom_hi)

    @staticmethod
    def _cos(start, end, pct):
        cos_out = np.cos(np.pi * pct) + 1.0
        return end + (start - end) / 2.0 * cos_out

    def _set(self, lr, beta1):
        for group in self.optimizer.param_groups:
            group['lr'] = float(lr)
            beta2 = group['betas'][1]
            group['betas'] = (float(beta1), beta2)

    def step(self):
        # FastAI Scheduler.step(): increment first, then evaluate n/n_iter.
        self.n += 1
        n_iter = max(1, self.phase1 if self.phase == 0 else self.phase2)
        pct = self.n / n_iter
        if self.phase == 0:
            lr = self._cos(self.low_lr, self.max_lr, pct)
            mom = self._cos(self.mom_hi, self.mom_lo, pct)
        else:
            lr = self._cos(self.max_lr, self.final_lr, pct)
            mom = self._cos(self.mom_lo, self.mom_hi, pct)
        self._set(lr, mom)
        if self.n >= n_iter and self.phase == 0:
            self.phase = 1
            self.n = 0



# ============================================================
# TRAINING
# ============================================================
def train_epoch(model, loader, optimizer, criterion,
                scheduler, device):
    """
    Single training epoch.
    One-cycle LR [Smith and Topin, 2019].
    BCEWithLogitsLoss for multi-label [Strodthoff et al., 2021].
    """
    model.train()
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)  # BCEWithLogitsLoss
        loss.backward()

        # FastAI v1 true_wd: decoupled weight decay using the CURRENT LR.
        # Applied immediately before the optimizer update.
        with torch.no_grad():
            for group in optimizer.param_groups:
                lr = group['lr']
                decay = 1.0 - WEIGHT_DECAY * lr
                for p in group['params']:
                    if p.grad is not None:
                        p.mul_(decay)

        optimizer.step()
        scheduler.step()  # One-cycle: step per batch
        total += loss.item()
    return total / len(loader)


def predict(model, loader, device):
    """
    Get sigmoid predictions for all records.
    Sigmoid applied at inference [Strodthoff et al., 2021].
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in loader:
            # Sigmoid for multi-label [Strodthoff et al., 2021]
            out = torch.sigmoid(model(x.to(device)))
            preds.extend(out.cpu().numpy())
    return np.array(preds)


# ============================================================
# MAIN — Run exp0 and exp3
# ============================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load PTB-XL CSV [Wagner et al., 2020]
    print("\nLoading ptbxl_database.csv...")
    Y = pd.read_csv(PTBXL_PATH + 'ptbxl_database.csv', index_col='ecg_id')
    # Adapted from Strodthoff et al. (2021), utils.py
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    print(f"Total records: {len(Y)}")

    # Load signals at 100 Hz [Strodthoff et al., 2021]
    X = load_raw_data_ptbxl(Y, SAMPLING_RATE, PTBXL_PATH)

    # Results storage
    all_results = {}

    # ── Run each experiment ──────────────────────────────────
    for exp_name, ctype in EXPERIMENTS.items():
        print(f"\n{'='*65}")
        print(f"EXPERIMENT: {exp_name} (task='{ctype}')")
        print(f"Target AUROC: {TARGETS[exp_name]}")
        print(f"{'='*65}")

        # Label aggregation [Strodthoff et al., 2021]
        Y_agg = compute_label_aggregations(Y.copy(), PTBXL_PATH, ctype)

        # Select data + multi-hot encoding [Strodthoff et al., 2021]
        X_sel, Y_sel, y, mlb = select_data(X, Y_agg, ctype)
        n_classes = y.shape[1]
        print(f"n_classes: {n_classes}")

        # Official fold splits [Strodthoff et al., 2021]
        X_train = X_sel[Y_sel.strat_fold <= 8]
        y_train = y[Y_sel.strat_fold <= 8]
        X_val   = X_sel[Y_sel.strat_fold == 9]
        y_val   = y[Y_sel.strat_fold == 9]
        X_test  = X_sel[Y_sel.strat_fold == 10]
        y_test  = y[Y_sel.strat_fold == 10]

        print(f"\nSplits:")
        print(f"  Train : {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

        # Normalisation [Strodthoff et al., 2021]
        X_train, X_val, X_test, ss = preprocess_signals(
            X_train, X_val, X_test
        )

        # DataLoaders
        # MOD-6: random crop for training [Strodthoff et al., 2021]
        tr_loader = DataLoader(
            ECGDataset(X_train, y_train,
                       crop_len=CROP_LEN, random_crop=True),
            batch_size=BATCH_SIZE, shuffle=True
        )
        # Val and test use full signal (crops handled in get_crop_predictions)
        val_loader  = DataLoader(ECGDataset(X_val,  y_val),
                                 batch_size=BATCH_SIZE)
        test_loader = DataLoader(ECGDataset(X_test, y_test),
                                 batch_size=BATCH_SIZE)

        # Model [Strodthoff et al., 2021]
        model = ResNet1d(
            input_channels=N_LEADS,
            num_classes=n_classes
        ).to(device)

        # Apply weight initialisation [Strodthoff et al., 2021]
        # MOD-5: consistent with weight_init() in fastai_model.py
        model.apply(weight_init)

        # Adam with FastAI v1 betas + manual true_wd [Strodthoff et al., 2021]
        # FastAI v1 source: AdamW = partial(optim.Adam, betas=(0.9, 0.99))
        # beta1 starts at 0.95 (max_momentum) and is cycled by OneCycleLR
        # beta2 = 0.99 confirmed from FastAI v1 source
        # weight_decay=0.01 confirmed from fastai_model.py wd=1e-2
        # [Loshchilov and Hutter, 2019]
        # FastAI v1 optimizer semantics: Adam + decoupled true_wd applied manually.
        # Internal optimizer weight_decay is zero; before optimizer.step(),
        # parameters are multiplied by (1 - current_lr * WEIGHT_DECAY).
        optimizer = optim.Adam(
            model.parameters(),
            lr=MAX_LR,
            betas=(0.95, 0.99),
            weight_decay=0.0
        )

        # BCEWithLogitsLoss — multi-label [Strodthoff et al., 2021]
        criterion = nn.BCEWithLogitsLoss()

        # Source-faithful FastAI-v1 two-phase OneCycle scheduler.
        # This intentionally replaces torch.optim.lr_scheduler.OneCycleLR
        # to reproduce fastai1 Scheduler rounding and end-of-batch stepping.
        scheduler = FastAIV1OneCycle(
            optimizer,
            total_steps=len(tr_loader) * EPOCHS,
            max_lr=MAX_LR,
            pct_start=0.3,
            div_factor=25.0,
            moms=(0.95, 0.85)
        )

        print(f"\nTraining for {EPOCHS} epochs...")
        print(f"Batch: {BATCH_SIZE} | One-cycle LR={MAX_LR} | BCE | Crop={CROP_LEN}")
        print("-" * 50)

        # Training — best-val checkpoint retained for DIAGNOSTIC only; not original selection protocol
        best_val_auroc = 0
        best_val_preds = None

        for ep in range(EPOCHS):
            loss = train_epoch(model, tr_loader, optimizer,
                               criterion, scheduler, device)

            # Validate — crop-based prediction [MOD-6]
            val_preds = get_crop_predictions(
                model, X_val, y_val,
                CROP_LEN, CROP_STRIDE, BATCH_SIZE, device
            )
            val_auroc = evaluate_experiment(y_val, val_preds)

            # Val-based checkpoint selection
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_val_preds = val_preds
                # Save best-validation checkpoint
                torch.save(
                    model.state_dict(),
                    f"scripts_final/best_resnet_{exp_name}.pth"
                )

            if (ep + 1) % 10 == 0 or ep == 0:
                print(f"Epoch {ep+1:3d}/{EPOCHS} | "
                      f"Loss: {loss:.4f} | "
                      f"Val AUROC: {val_auroc:.4f}")

        # Save final-epoch checkpoint (FastAI default behaviour)
        torch.save(
            model.state_dict(),
            f"scripts_final/final_resnet_{exp_name}.pth"
        )

        # ── Protocol A: Best-validation checkpoint ──────────────
        model.load_state_dict(
            torch.load(f"scripts_final/best_resnet_{exp_name}.pth",
                       map_location=device)
        )
        test_preds_bestval = get_crop_predictions(
            model, X_test, y_test,
            CROP_LEN, CROP_STRIDE, BATCH_SIZE, device
        )
        test_auroc_bestval = evaluate_experiment(y_test, test_preds_bestval)

        # ── Protocol B: Final-epoch checkpoint (FastAI default) ──
        model.load_state_dict(
            torch.load(f"scripts_final/final_resnet_{exp_name}.pth",
                       map_location=device)
        )
        test_preds_final = get_crop_predictions(
            model, X_test, y_test,
            CROP_LEN, CROP_STRIDE, BATCH_SIZE, device
        )
        test_auroc_final = evaluate_experiment(y_test, test_preds_final)

        # Source-faithful primary protocol: the original FastAI implementation
        # saves/evaluates the model after fit_one_cycle completes (final epoch).
        # Best-validation checkpoint is retained only as a diagnostic comparison.
        test_preds = test_preds_final
        test_auroc = test_auroc_final

        # Bootstrap CI [Efron & Tibshirani, 1993]
        print(f"\nComputing Bootstrap CI ({N_BOOTSTRAP} iterations)...")
        point, mean, lower, upper = compute_bootstrap_ci(
            y_test, test_preds, N_BOOTSTRAP
        )

        # Results
        print(f"\n{'='*65}")
        print(f"RESULTS — {exp_name} (task='{ctype}')")
        print(f"{'='*65}")
        print(f"  Best Val AUROC       : {best_val_auroc:.4f}")
        print(f"  Test AUROC (best-val): {test_auroc_bestval:.4f}  ← diagnostic only")
        print(f"  Test AUROC (final ep): {test_auroc_final:.4f}  ← PRIMARY/source-faithful")
        print(f"  Bootstrap Mean  : {mean:.4f}")
        print(f"  95% CI          : [{lower:.4f}, {upper:.4f}]")
        print(f"  Target          : {TARGETS[exp_name]:.4f}")
        print(f"  Difference      : {(test_auroc - TARGETS[exp_name])*100:+.2f}%")

        all_results[exp_name] = {
            'task':                  ctype,
            'n_classes':             n_classes,
            'best_val_auroc':        round(best_val_auroc, 4),
            'test_auroc_bestval':    round(test_auroc_bestval, 4),
            'test_auroc_final':      round(test_auroc_final, 4),
            'test_auroc':            round(test_auroc_final, 4),
            'bootstrap_mean': round(mean,  4),
            'ci_lower':       round(lower, 4),
            'ci_upper':       round(upper, 4),
            'target':         TARGETS[exp_name],
            'difference_pct': round((test_auroc - TARGETS[exp_name])*100, 2),
        }

    # Final summary
    print(f"\n{'='*65}")
    print("FINAL SUMMARY")
    print(f"{'='*65}")
    for exp, res in all_results.items():
        status = "✅ PASS" if abs(res['difference_pct']) <= 1.0 else "❌ FAIL"
        print(f"\n{exp} ({res['task']}):")
        print(f"  Test AUROC : {res['test_auroc']:.4f} "
              f"[95% CI: {res['ci_lower']:.4f}–{res['ci_upper']:.4f}]")
        print(f"  Target     : {res['target']:.4f}")
        print(f"  Difference : {res['difference_pct']:+.2f}%  {status}")

    # Save results
    results_df = pd.DataFrame(all_results).T
    results_df.to_csv(OUTPUT_DIR + 'reproduction_results.csv')
    print(f"\nResults saved to: {OUTPUT_DIR}reproduction_results.csv")
