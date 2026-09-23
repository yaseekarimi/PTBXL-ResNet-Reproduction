"""
v6-EVAL diagnostic for exp3 (rhythm). NO TRAINING.
Place beside resnet_exp0_exp3_v6_fastai_sourcefaithful_v101.py in scripts_final.
"""

import os, sys, ast
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import resnet_exp0_exp3_v6_fastai_sourcefaithful_v101 as v6

EXP_NAME, CTYPE = "exp3", "rhythm"
OUT = "results/exp_reproduction/v6_eval_diagnostic"
os.makedirs(OUT, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 76)
print("v6-EVAL DIAGNOSTIC — exp3 — NO TRAINING")
print("=" * 76)
print("Device:", device)

Y = pd.read_csv(v6.PTBXL_PATH + "ptbxl_database.csv", index_col="ecg_id")
Y.scp_codes = Y.scp_codes.apply(ast.literal_eval)
X = v6.load_raw_data_ptbxl(Y, v6.SAMPLING_RATE, v6.PTBXL_PATH)
Y_agg = v6.compute_label_aggregations(Y.copy(), v6.PTBXL_PATH, CTYPE)
X_sel, Y_sel, y, mlb = v6.select_data(X, Y_agg, CTYPE)
classes = np.asarray(mlb.classes_, dtype=str)

X_train = X_sel[Y_sel.strat_fold <= 8]
X_val   = X_sel[Y_sel.strat_fold == 9]
X_test  = X_sel[Y_sel.strat_fold == 10]
y_train = y[Y_sel.strat_fold <= 8]
y_val   = y[Y_sel.strat_fold == 9]
y_test  = y[Y_sel.strat_fold == 10]

X_train, X_val, X_test, _ = v6.preprocess_signals(X_train, X_val, X_test)

pos = y_test.sum(axis=0).astype(int)
neg = len(y_test) - pos

print("\nDATA / LABEL AUDIT")
print("Metadata records :", len(Y))
print("Selected exp3    :", len(X_sel))
print("Train/Val/Test   :", len(X_train), len(X_val), len(X_test))
print("Classes          :", classes.tolist())
print("X_test shape     :", X_test.shape)
print("y_test shape     :", y_test.shape)
print("NaN X/y          :", bool(np.isnan(X_test).any()), bool(np.isnan(y_test).any()))

print("\nTEST CLASS COUNTS")
for c, p, n in zip(classes, pos, neg):
    print(f"{c:10s} positive={p:4d} negative={n:4d}")

def predict_with_crop_audit(model, X):
    model.eval()
    out = np.zeros((len(X), len(classes)), dtype=np.float32)
    counts = []
    with torch.no_grad():
        for i, sig in enumerate(X):
            starts = list(range(0, sig.shape[0] - v6.CROP_LEN + 1,
                                v6.CROP_STRIDE))
            if not starts:
                starts = [0]
            crops = np.stack(
                [sig[s:s+v6.CROP_LEN, :].T for s in starts]
            ).astype(np.float32)
            cp = torch.sigmoid(
                model(torch.from_numpy(crops).to(device))
            ).cpu().numpy()
            out[i] = cp.max(axis=0)
            counts.append(len(starts))
    return out, counts

def evaluate_checkpoint(label, path):
    print("\n" + "=" * 76)
    print("CHECKPOINT:", label)
    print("=" * 76)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    model = v6.ResNet1d(v6.N_LEADS, len(classes)).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    preds, counts = predict_with_crop_audit(model, X_test)

    print("Prediction shape :", preds.shape)
    print("Crops/record     :", sorted(set(counts)))
    print("Total crops      :", sum(counts))
    print("NaN predictions  :", bool(np.isnan(preds).any()))
    print("Prediction range :", float(preds.min()), float(preds.max()))

    per_class = []
    print("\nPER-CLASS AUROC")
    for j, c in enumerate(classes):
        auc = roc_auc_score(y_test[:, j], preds[:, j])
        per_class.append(auc)
        print(f"{c:10s}: {auc:.6f} "
              f"(positive={int(pos[j])}, negative={int(neg[j])})")

    a = roc_auc_score(y_test, preds, average="macro")
    b = v6.evaluate_experiment(y_test, preds)
    c = float(np.mean(per_class))
    print("\nMacro sklearn :", f"{a:.10f}")
    print("Macro v6      :", f"{b:.10f}")
    print("Macro manual  :", f"{c:.10f}")
    print("Agreement     :", "PASS" if np.allclose([a,b],[b,c], atol=1e-12) else "FAIL")

    stem = label.lower().replace("-", "_").replace(" ", "_")
    np.save(os.path.join(OUT, f"preds_{stem}.npy"), preds)
    np.save(os.path.join(OUT, f"targs_{stem}.npy"), y_test)
    np.save(os.path.join(OUT, f"classes_{stem}.npy"), classes)
    pd.DataFrame({
        "class": classes, "positive": pos, "negative": neg, "auroc": per_class
    }).to_csv(os.path.join(OUT, f"per_class_{stem}.csv"), index=False)
    return a

best = evaluate_checkpoint(
    "best_val", f"scripts_final/best_resnet_{EXP_NAME}.pth"
)
final = evaluate_checkpoint(
    "final_epoch", f"scripts_final/final_resnet_{EXP_NAME}.pth"
)

print("\n" + "=" * 76)
print("SUMMARY")
print("=" * 76)
print(f"Best-val   : {best:.10f}")
print(f"Final epoch: {final:.10f}")
print("Published  : 0.9460000000")
print("Saved to  :", OUT)
print("NO TRAINING was performed.")
