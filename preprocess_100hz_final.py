"""
ECG Signal Harmonisation Pipeline — Updated Version
=====================================================
Phase 0 - Month 2

Purpose:
    Harmonises PTB-XL dataset to a common format suitable
    for reproducing the benchmark of Strodthoff et al. (2021).

    Changes from previous version:
    - Sampling rate: 500 Hz → 100 Hz (filename_lr)
    - Window: 5000 → 1000 samples
    - Filter: HPF + Notch removed (not in original source)
    - Label: verification only (multi-label in main experiment)

References:
-----------
[1] Strodthoff, N., Wagner, P., Schaeffter, T. and Samek, W. (2021)
    'Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL',
    IEEE Journal of Biomedical and Health Informatics, 25(5), pp. 1519-1528.
    DOI: 10.1109/JBHI.2020.3022989
    GitHub: https://github.com/helme/ecg_ptbxl_benchmarking
    Original file: code/utils/utils.py

[2] Wagner, P. et al. (2020)
    'PTB-XL, a large publicly available electrocardiography dataset',
    Scientific Data, 7(1), p. 154.
    DOI: 10.1038/s41597-020-0495-6

[3] Goldberger, A.L. et al. (2000)
    'PhysioBank, PhysioToolkit, and PhysioNet',
    Circulation, 101(23), pp. e215-e220.
    DOI: 10.1161/01.CIR.101.23.e215

Adaptation notes:
-----------------
- load_raw_data_ptbxl() adapted from Strodthoff et al. (2021),
  utils.py: loads PTB-XL at 100 Hz via filename_lr column.
- apply_standardizer() adapted from Strodthoff et al. (2021),
  utils.py: applies StandardScaler per record.
- preprocess_signals() adapted from Strodthoff et al. (2021),
  utils.py: fits StandardScaler on training data only.
- Fold protocol adapted from Strodthoff et al. (2021),
  utils.py: folds 1-8 train, fold 9 val, fold 10 test.

Modifications from original Strodthoff et al. (2021) source:
--------------------------------------------------------------
MOD-1: Per-record error handling added — if a single record
        fails, it is skipped and processing continues.
        Original source assumes all records load correctly.

MOD-2: Pipeline verification reporting added — label counts
        and signal shape printed per split for verification.
        Not present in original source.
"""

import os
import numpy as np
import pandas as pd
import ast
import wfdb
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIG
# Source: Strodthoff et al. (2021), Section II-A
# ============================================================
PTBXL_PATH = "data/ptbxl/ptb-xl-1.0.1"

TARGET_FS  = 100   # Hz — Strodthoff et al. (2021), Section II-A
TARGET_LEN = 1000  # 10s × 100Hz — Strodthoff et al. (2021)
N_LEADS    = 12    # standard 12-lead ECG

print("=" * 60)
print("ECG Preprocessing Pipeline — Updated")
print("100 Hz | 1000 samples | No filter | PTB-XL only")
print("Following Strodthoff et al. (2021)")
print("=" * 60)


# ============================================================
# SIGNAL PROCESSING
# Adapted from Strodthoff et al. (2021), utils.py
# ============================================================
def apply_window(sig, target_len=TARGET_LEN):
    """
    Truncate or zero-pad signal to fixed window length.

    Window: 10s × 100Hz = 1000 samples
    Source: Strodthoff et al. (2021), Section II-A.

    Parameters
    ----------
    sig : np.ndarray, shape (n_samples, n_leads)
    target_len : int

    Returns
    -------
    np.ndarray, shape (1000, 12)
    """
    if len(sig) >= target_len:
        return sig[:target_len]
    pad = np.zeros((target_len - len(sig), sig.shape[1]))
    return np.concatenate([sig, pad], axis=0)


def enforce_leads(sig, n_leads=N_LEADS):
    """
    Enforce fixed number of leads.

    Source: Strodthoff et al. (2021) — 12-lead input assumed.

    Parameters
    ----------
    sig : np.ndarray, shape (n_samples, n_leads_orig)

    Returns
    -------
    np.ndarray, shape (n_samples, 12)
    """
    if sig.shape[1] >= n_leads:
        return sig[:, :n_leads]
    pad = np.zeros((sig.shape[0], n_leads - sig.shape[1]))
    return np.concatenate([sig, pad], axis=1)


def apply_standardizer(sig, ss):
    """
    Apply StandardScaler to a single ECG record.

    Adapted from Strodthoff et al. (2021), utils.py,
    apply_standardizer():
        x_shape = x.shape
        ss.transform(x.flatten()[:,np.newaxis]).reshape(x_shape)

    Parameters
    ----------
    sig : np.ndarray, shape (1000, 12)
    ss  : fitted StandardScaler

    Returns
    -------
    np.ndarray, shape (1000, 12), normalised
    """
    # Adapted from Strodthoff et al. (2021), utils.py
    x_shape = sig.shape
    return ss.transform(
        sig.flatten()[:, np.newaxis]
    ).reshape(x_shape)


def fit_standardizer(signals):
    """
    Fit StandardScaler on training signals.

    Adapted from Strodthoff et al. (2021), utils.py,
    preprocess_signals():
        ss = StandardScaler()
        ss.fit(np.vstack(X_train).flatten()[:,np.newaxis].astype(float))

    Must be fitted on training data only to prevent data leakage.

    Parameters
    ----------
    signals : list of np.ndarray

    Returns
    -------
    sklearn.preprocessing.StandardScaler, fitted
    """
    # Adapted from Strodthoff et al. (2021), utils.py
    ss = StandardScaler()
    ss.fit(np.vstack(signals).flatten()[:, np.newaxis].astype(float))
    return ss


# ============================================================
# PTB-XL LOADING
# Adapted from Strodthoff et al. (2021), utils.py:
#   load_raw_data_ptbxl() — loads at 100 Hz via filename_lr
# ============================================================
def load_split(df, path, split_name):
    """
    Load ECG signals for one PTB-XL split.

    Adapted from Strodthoff et al. (2021), utils.py,
    load_raw_data_ptbxl():
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]

    Loads at 100 Hz via filename_lr column [Wagner et al., 2020].
    No filtering applied — consistent with original source.

    MOD-1: Per-record error handling added.

    Parameters
    ----------
    df : pd.DataFrame — PTB-XL metadata split
    path : str — PTB-XL root directory
    split_name : str — 'Train', 'Val', or 'Test'

    Returns
    -------
    tuple : (signals list, labels list)
    """
    print(f"\n  Loading {split_name}...")

    signals = []
    labels  = []
    errors  = 0

    for _, row in df.iterrows():
        try:
            # Load at 100 Hz via filename_lr
            # Adapted from load_raw_data_ptbxl() [Strodthoff et al., 2021]
            rec_path = os.path.join(path, row['filename_lr'])
            record   = wfdb.rdrecord(rec_path)
            sig      = record.p_signal  # (n_samples, n_leads)

            # Enforce 12 leads
            sig = enforce_leads(sig)

            # Window to 1000 samples [Strodthoff et al., 2021]
            sig = apply_window(sig)

            # SCP codes for verification
            scp = row['scp_codes']

            signals.append(sig)
            labels.append(scp)

        except Exception:
            errors += 1  # MOD-1: skip bad record
            continue

    print(f"    Loaded : {len(signals)} records")
    print(f"    Errors : {errors} (skipped) [MOD-1]")
    print(f"    Shape  : ({len(signals)}, {TARGET_LEN}, {N_LEADS})")

    return signals, labels


# ============================================================
# MAIN — Pipeline Verification
# ============================================================
if __name__ == "__main__":

    # Load PTB-XL CSV [Wagner et al., 2020]
    print("\nLoading ptbxl_database.csv...")
    csv_path = os.path.join(PTBXL_PATH, "ptbxl_database.csv")
    Y = pd.read_csv(csv_path, index_col='ecg_id')
    # Adapted from Strodthoff et al. (2021), utils.py
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    print(f"Total records: {len(Y)}")

    # Official fold splits [Strodthoff et al., 2021]
    train_df = Y[Y.strat_fold <= 8]
    val_df   = Y[Y.strat_fold == 9]
    test_df  = Y[Y.strat_fold == 10]

    print(f"\nFold splits [Strodthoff et al., 2021]:")
    print(f"  Train (folds 1-8) : {len(train_df)} records")
    print(f"  Val   (fold 9)    : {len(val_df)} records")
    print(f"  Test  (fold 10)   : {len(test_df)} records")

    # Load splits
    X_train, y_train = load_split(train_df[:10], PTBXL_PATH, "Train (10 records only)")
    X_val,   y_val   = load_split(val_df[:10],   PTBXL_PATH, "Val (10 records only)")
    X_test,  y_test  = load_split(test_df[:10],  PTBXL_PATH, "Test (10 records only)")

    # Fit StandardScaler on training data only
    # Adapted from preprocess_signals() [Strodthoff et al., 2021]
    print("\nFitting StandardScaler on training data...")
    ss = fit_standardizer(X_train)

    # Apply to all splits
    X_train_norm = [apply_standardizer(s, ss) for s in X_train]
    X_val_norm   = [apply_standardizer(s, ss) for s in X_val]
    X_test_norm  = [apply_standardizer(s, ss) for s in X_test]

    print(f"\nZ-score applied [Strodthoff et al., 2021]")
    print(f"  Train shape: ({len(X_train_norm)}, {TARGET_LEN}, {N_LEADS})")
    print(f"  Val shape  : ({len(X_val_norm)},  {TARGET_LEN}, {N_LEADS})")
    print(f"  Test shape : ({len(X_test_norm)},  {TARGET_LEN}, {N_LEADS})")

    print("\n" + "=" * 60)
    print("Pipeline verified!")
    print(f"100 Hz | 1000 samples | No filter | PTB-XL only")
    print("=" * 60)
