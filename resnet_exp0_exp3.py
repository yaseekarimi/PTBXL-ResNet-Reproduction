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
"""

import os
import ast
import pickle
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
PTBXL_PATH = "data/ptbxl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
OUTPUT_DIR  = "results/exp_reproduction/"

SAMPLING_RATE = 100   # Hz — Strodthoff et al. (2021)
TARGET_LEN    = 1000  # 10s × 100Hz
N_LEADS       = 12

BATCH_SIZE = 128      # Strodthoff et al. (2021)
EPOCHS     = 100
MAX_LR     = 0.001    # One-cycle — Smith and Topin (2019)
N_BOOTSTRAP = 100     # Bootstrap iterations — Efron & Tibshirani (1993)

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
# PYTORCH DATASET
# ============================================================
class ECGDataset(Dataset):
    def __init__(self, X, y):
        # Transpose to (n, 12, 1000) for Conv1d
        self.X = torch.tensor(
            X.transpose(0, 2, 1), dtype=torch.float32
        )
        self.y = torch.tensor(y, dtype=torch.float32)  # float for BCE

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


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

        # Head — MOD-1: native PyTorch replaces fastai create_head1d
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(inplanes, num_classes)
            # No sigmoid here — BCEWithLogitsLoss handles it
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
        return self.head(x)


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
        tr_loader   = DataLoader(ECGDataset(X_train, y_train),
                                 batch_size=BATCH_SIZE, shuffle=True)
        val_loader  = DataLoader(ECGDataset(X_val,   y_val),
                                 batch_size=BATCH_SIZE)
        test_loader = DataLoader(ECGDataset(X_test,  y_test),
                                 batch_size=BATCH_SIZE)

        # Model [Strodthoff et al., 2021]
        model = ResNet1d(
            input_channels=N_LEADS,
            num_classes=n_classes
        ).to(device)

        # Adam [Kingma and Ba, 2015]
        optimizer = optim.Adam(model.parameters(), lr=MAX_LR)

        # BCEWithLogitsLoss — multi-label [Strodthoff et al., 2021]
        criterion = nn.BCEWithLogitsLoss()

        # One-cycle LR [Smith and Topin, 2019]
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=MAX_LR,
            epochs=EPOCHS,
            steps_per_epoch=len(tr_loader)
        )

        print(f"\nTraining for {EPOCHS} epochs...")
        print(f"Batch: {BATCH_SIZE} | One-cycle LR | BCE loss")
        print("-" * 50)

        # Training — Val-based checkpoint [MOD — not in original]
        best_val_auroc = 0
        best_val_preds = None

        for ep in range(EPOCHS):
            loss = train_epoch(model, tr_loader, optimizer,
                               criterion, scheduler, device)

            # Validate
            val_preds  = predict(model, val_loader,  device)
            val_auroc  = evaluate_experiment(y_val, val_preds)

            # Val-based checkpoint selection
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_val_preds = val_preds
                # Save best model checkpoint
                torch.save(
                    model.state_dict(),
                    f"scripts_final/best_resnet_{exp_name}.pth"
                )

            if (ep + 1) % 10 == 0 or ep == 0:
                print(f"Epoch {ep+1:3d}/{EPOCHS} | "
                      f"Loss: {loss:.4f} | "
                      f"Val AUROC: {val_auroc:.4f}")

        # Load best checkpoint for test evaluation
        model.load_state_dict(
            torch.load(f"scripts_final/best_resnet_{exp_name}.pth",
                       map_location=device)
        )

        # Single test evaluation — Val-selected checkpoint
        test_preds = predict(model, test_loader, device)
        test_auroc = evaluate_experiment(y_test, test_preds)

        # Bootstrap CI [Efron & Tibshirani, 1993]
        print(f"\nComputing Bootstrap CI ({N_BOOTSTRAP} iterations)...")
        point, mean, lower, upper = compute_bootstrap_ci(
            y_test, test_preds, N_BOOTSTRAP
        )

        # Results
        print(f"\n{'='*65}")
        print(f"RESULTS — {exp_name} (task='{ctype}')")
        print(f"{'='*65}")
        print(f"  Best Val AUROC  : {best_val_auroc:.4f}")
        print(f"  Test AUROC      : {test_auroc:.4f}")
        print(f"  Bootstrap Mean  : {mean:.4f}")
        print(f"  95% CI          : [{lower:.4f}, {upper:.4f}]")
        print(f"  Target          : {TARGETS[exp_name]:.4f}")
        print(f"  Difference      : {(test_auroc - TARGETS[exp_name])*100:+.2f}%")

        all_results[exp_name] = {
            'task':           ctype,
            'n_classes':      n_classes,
            'best_val_auroc': round(best_val_auroc, 4),
            'test_auroc':     round(test_auroc, 4),
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
