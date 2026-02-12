
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Omics modality-specific predictive models for glycaemic response (2h iAUC).

Notes on evaluation:
- Train/test split is performed at the participant level (GroupShuffleSplit) to avoid within-participant overlap.
- Models should only use covariates/features available at baseline or at the corresponding time point.

Pipeline (matching your Methods text):
1) For each omics modality: build a separate predictive model.
2) Predictors = modality features + baseline clinical covariates (age, sex, BMI, fasting glucose, meds, ...).
3) Feature selection via LightGBM: feature importance quantified by the frequency a feature is used for node splitting across trees
   (tree-usage frequency). Low-frequency features are excluded.
4) Final model training via XGBoost (regression).
5) Hyperparameter optimization via Optuna.
6) Feature stability / importance estimated via resampling across multiple random seeds.
7) Evaluation: Pearson, MAE, RMSE + calibration plots (regression reliability diagram).
8) Optional: cross-population generalizability (train on subgroup A, test on subgroup B) and longitudinal robustness (day-wise metrics).

This file is intentionally self-contained for GitHub (single script).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

import lightgbm as lgb
from xgboost import XGBRegressor
import optuna


# -----------------------------
# Plot defaults (Adobe-editable PDF)
# -----------------------------
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"]  = 42
mpl.rcParams["font.family"]  = "Arial"


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int) -> None:
    np.random.seed(seed)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 2:
        return np.nan, np.nan
    if np.std(y_true[m]) == 0 or np.std(y_pred[m]) == 0:
        return np.nan, np.nan
    r, p = pearsonr(y_true[m], y_pred[m])
    return float(r), float(p)


# -----------------------------
# Calibration plot (regression)
# -----------------------------
def regression_calibration_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> pd.DataFrame:
    """
    Binned calibration for regression:
    - Bin by y_pred (quantile or uniform)
    - For each bin, compute mean_pred and mean_true

    Returns DataFrame with: bin, n, mean_pred, mean_true
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[m]
    y_pred = y_pred[m]
    if len(y_true) == 0:
        return pd.DataFrame(columns=["bin", "n", "mean_pred", "mean_true"])

    if strategy not in {"quantile", "uniform"}:
        raise ValueError("strategy must be 'quantile' or 'uniform'")

    if strategy == "quantile":
        # duplicates='drop' avoids errors when many equal predictions
        bins = pd.qcut(y_pred, q=n_bins, duplicates="drop")
    else:
        bins = pd.cut(y_pred, bins=n_bins)

    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "bin": bins})
    out = (
        df.groupby("bin", observed=True)
          .agg(n=("y_true", "size"),
               mean_pred=("y_pred", "mean"),
               mean_true=("y_true", "mean"))
          .reset_index(drop=True)
    )
    out.insert(0, "bin", np.arange(len(out)))
    return out


def plot_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_pdf: str,
    title: str = "Calibration (regression)",
    n_bins: int = 10,
    strategy: str = "quantile",
    color: str = "#2C74B6",
) -> None:
    cal = regression_calibration_curve(y_true, y_pred, n_bins=n_bins, strategy=strategy)
    if cal.empty:
        return

    lo = min(cal["mean_pred"].min(), cal["mean_true"].min())
    hi = max(cal["mean_pred"].max(), cal["mean_true"].max())
    pad = (hi - lo) * 0.05 if hi > lo else 1.0
    lo, hi = lo - pad, hi + pad

    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.plot([lo, hi], [lo, hi], linewidth=1.0)
    ax.plot(cal["mean_pred"], cal["mean_true"], marker="o", linewidth=1.5, color=color)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Mean predicted")
    ax.set_ylabel("Mean observed (true)")
    ax.set_title(title)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()


# -----------------------------
# LightGBM: split-usage frequency feature selection
# -----------------------------
def _lgb_params(
    learning_rate=0.01,
    min_data_in_leaf=60,
    bagging_fraction=0.8,
    feature_fraction=0.6,
    max_depth=-1,
    metric="l2",
    objective="regression",
    verbose=-1,
    **kwargs,
) -> dict:
    p = dict(
        objective=objective,
        metric=metric,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_data_in_leaf=min_data_in_leaf,
        bagging_fraction=bagging_fraction,
        bagging_freq=1,
        feature_fraction=feature_fraction,
        verbose=verbose,
    )
    p.update(kwargs)
    return p


def _trees_feature_usage_counts(booster: lgb.Booster, n_features: int) -> np.ndarray:
    """
    Parse LightGBM model dump and count:
    for each feature, in how many trees it appears at least once in a split.
    """
    model_dump = booster.dump_model()
    counts = np.zeros(n_features, dtype=int)

    for tree in model_dump.get("tree_info", []):
        used = set()

        def dfs(node):
            if "split_feature" in node:
                used.add(int(node["split_feature"]))
                dfs(node["left_child"])
                dfs(node["right_child"])

        dfs(tree["tree_structure"])
        for j in used:
            counts[j] += 1

    return counts


def select_features_by_tree_usage(
    X: pd.DataFrame,
    y: pd.Series,
    groups: Optional[Iterable] = None,
    n_splits: int = 1,
    n_estimators: int = 2000,
    learning_rate: float = 0.01,
    subsample: float = 0.8,
    feature_fraction: float = 0.6,
    min_data_in_leaf: int = 60,
    top_k: Optional[int] = 300,
    stable_frac: float = 0.6,
    random_state: int = 42,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Returns:
      selected_features: list[str]
      details: DataFrame with used_trees_total, used_trees_perc, chosen_freq
    """
    X = X.copy()
    y = y.copy()

    # Ensure numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    feat_names = list(map(str, X.columns))
    n_features = X.shape[1]

    params = _lgb_params(
        learning_rate=learning_rate,
        min_data_in_leaf=min_data_in_leaf,
        bagging_fraction=subsample,
        feature_fraction=feature_fraction,
    )

    if n_splits <= 1:
        dtrain = lgb.Dataset(X.values, label=y.values, feature_name=feat_names)
        booster = lgb.train(params, dtrain, num_boost_round=n_estimators)
        counts = _trees_feature_usage_counts(booster, n_features)
        details = pd.DataFrame(
            {
                "used_trees_total": counts,
                "used_trees_perc": counts / float(n_estimators),
                "chosen_freq": 1.0,
            },
            index=feat_names,
        ).sort_values("used_trees_total", ascending=False)

        if top_k is None:
            selected = details.index.tolist()
        else:
            selected = details.index[: int(top_k)].tolist()
        return selected, details

    # Simple resampling CV: repeated train subsamples by groups if provided
    # (kept intentionally simple for single-file reproducibility)
    rng = np.random.RandomState(random_state)
    counts_total = np.zeros(n_features, dtype=int)
    chosen_counter = np.zeros(n_features, dtype=int)
    effective = 0

    idx_all = np.arange(X.shape[0])
    if groups is not None:
        groups = np.asarray(list(groups))
        uniq = np.unique(groups)
    else:
        uniq = None

    for _ in range(n_splits):
        if uniq is not None:
            tr_groups = rng.choice(uniq, size=int(len(uniq) * 0.8), replace=False)
            tr_mask = np.isin(groups, tr_groups)
            tr_idx = idx_all[tr_mask]
        else:
            tr_idx = rng.choice(idx_all, size=int(len(idx_all) * 0.8), replace=False)

        Xtr = X.iloc[tr_idx].values
        ytr = y.iloc[tr_idx].values
        dtr = lgb.Dataset(Xtr, label=ytr, feature_name=feat_names)
        booster = lgb.train(params, dtr, num_boost_round=n_estimators)
        counts = _trees_feature_usage_counts(booster, n_features)

        counts_total += counts
        effective += 1

        if top_k is not None:
            chosen_idx = np.argsort(-counts)[: int(top_k)]
            chosen_counter[chosen_idx] += 1

    details = pd.DataFrame(
        {
            "used_trees_total": counts_total,
            "used_trees_perc": counts_total / float(n_estimators * effective),
            "chosen_freq": chosen_counter / float(effective) if top_k is not None else np.nan,
        },
        index=feat_names,
    ).sort_values(["chosen_freq", "used_trees_total"], ascending=False)

    if top_k is None:
        selected = details.index.tolist()
    else:
        # stability filter first, then take top_k
        stable = details[details["chosen_freq"] >= stable_frac]
        if len(stable) == 0:
            stable = details
        selected = stable.index[: int(top_k)].tolist()

    return selected, details


# -----------------------------
# XGBoost + Optuna
# -----------------------------
def tune_xgb_with_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 200,
    seed: int = 42,
    direction: str = "maximize",  # maximize Pearson by default
) -> dict:
    X_train = X_train.apply(pd.to_numeric, errors="coerce")
    X_val = X_val.apply(pd.to_numeric, errors="coerce")

    y_train = np.asarray(y_train, dtype=float)
    y_val = np.asarray(y_val, dtype=float)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 8.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        }

        model = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
            **params,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        y_pred = model.predict(X_val)
        r, _ = safe_pearson(y_val, y_pred)
        # fallback: if Pearson is nan (zero var), maximize negative RMSE
        if not np.isfinite(r):
            return -rmse(y_val, y_pred) if direction == "maximize" else rmse(y_val, y_pred)
        return r if direction == "maximize" else -r

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    return study.best_params


# -----------------------------
# Data I/O
# -----------------------------
def load_table(path: str, index_col: Optional[str] = None) -> pd.DataFrame:
    if path.endswith(".parquet") or path.endswith(".pq"):
        df = pd.read_parquet(path)
    elif path.endswith(".tsv") or path.endswith(".txt"):
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)
    if index_col is not None and index_col in df.columns:
        df = df.set_index(index_col)
    return df


def align_on_key(
    labels: pd.DataFrame,
    covars: pd.DataFrame,
    feats: pd.DataFrame,
    key: str,
) -> pd.DataFrame:
    """
    Merge labels + covars + features on a shared key column (e.g., case_id / meal_ID).
    """
    for df in (labels, covars, feats):
        if key not in df.columns:
            raise ValueError(f"Missing key column '{key}' in one of the inputs.")
    out = labels.merge(covars, on=key, how="inner").merge(feats, on=key, how="inner")
    return out


# -----------------------------
# Train/Eval orchestration
# -----------------------------
@dataclass
class TrainConfig:
    key_col: str = "case_id"
    target_col: str = "iAUC_2h_true_baseline_sub_pos_dx5"
    group_col: str = "subject_id"   # held-out participants (participant-level split)
    day_col: Optional[str] = "day_index"
    ethnicity_col: Optional[str] = None

    test_size: float = 0.2
    seed: int = 42

    # LightGBM selection
    lgb_n_splits: int = 5
    lgb_n_estimators: int = 2000
    lgb_top_k: int = 300

    # Optuna
    optuna_trials: int = 200

    # Stability (XGB training with multiple seeds)
    stability_seeds: int = 20

    # Calibration plot
    calib_bins: int = 10
    calib_strategy: str = "quantile"


def split_train_test_by_group(df: pd.DataFrame, group_col: str, test_size: float, seed: int):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    groups = df[group_col].astype(str).values
    idx = np.arange(len(df))
    tr_idx, te_idx = next(gss.split(idx, groups=groups))
    return tr_idx, te_idx


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    r, p = safe_pearson(y_true, y_pred)
    return {
        "Pearson_r": r,
        "Pearson_p": p,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
    }


def run_one_modality(
    modality_name: str,
    labels_df: pd.DataFrame,
    covars_df: pd.DataFrame,
    feats_df: pd.DataFrame,
    outdir: str,
    cfg: TrainConfig,
    color: str = "#2C74B6",
) -> Dict[str, pd.DataFrame]:
    os.makedirs(outdir, exist_ok=True)

    df = align_on_key(labels_df, covars_df, feats_df, cfg.key_col)

    # Drop rows with missing target or group
    df = df[df[cfg.target_col].notna() & df[cfg.group_col].notna()].copy()

    tr_idx, te_idx = split_train_test_by_group(df, cfg.group_col, cfg.test_size, cfg.seed)
    df_tr = df.iloc[tr_idx].copy()
    df_te = df.iloc[te_idx].copy()

    # Prepare matrices
    feature_cols = [c for c in df.columns if c not in {
        cfg.target_col, cfg.group_col, cfg.key_col,
        *( [cfg.day_col] if cfg.day_col else [] ),
    }]

    X_tr_all = df_tr[feature_cols]
    y_tr_all = df_tr[cfg.target_col].astype(float)

    X_te_all = df_te[feature_cols]
    y_te_all = df_te[cfg.target_col].astype(float)

    # Inner split for Optuna validation (by group)
    tr2_idx, va_idx = split_train_test_by_group(df_tr, cfg.group_col, test_size=0.2, seed=cfg.seed)
    df_tr2 = df_tr.iloc[tr2_idx]
    df_va  = df_tr.iloc[va_idx]

    X_tr2 = df_tr2[feature_cols]
    y_tr2 = df_tr2[cfg.target_col].astype(float)
    X_va  = df_va[feature_cols]
    y_va  = df_va[cfg.target_col].astype(float)

    # 1) LightGBM feature selection (tree split usage)
    selected_features, fs_details = select_features_by_tree_usage(
        X_tr2, y_tr2,
        groups=df_tr2[cfg.group_col].astype(str).values,
        n_splits=cfg.lgb_n_splits,
        n_estimators=cfg.lgb_n_estimators,
        top_k=cfg.lgb_top_k,
        random_state=cfg.seed,
    )

    # Filter to selected features
    X_tr2_sel = X_tr2[selected_features]
    X_va_sel  = X_va[selected_features]
    X_tr_all_sel = X_tr_all[selected_features]
    X_te_sel     = X_te_all[selected_features]

    # 2) Optuna tune XGB on (train2 -> val)
    best_params = tune_xgb_with_optuna(
        X_tr2_sel, y_tr2, X_va_sel, y_va,
        n_trials=cfg.optuna_trials,
        seed=cfg.seed,
        direction="maximize",
    )

    final_params = dict(
        **best_params,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        random_state=cfg.seed,
        seed=cfg.seed,
    )

    # 3) Train final model on all training participants
    model = XGBRegressor(**final_params)
    model.fit(X_tr_all_sel.apply(pd.to_numeric, errors="coerce"), y_tr_all.values)

    # 4) Predict on test
    y_pred_te = model.predict(X_te_sel.apply(pd.to_numeric, errors="coerce"))

    metrics = evaluate_regression(y_te_all.values, y_pred_te)

    # Save predictions
    pred_df = df_te[[cfg.key_col, cfg.group_col]].copy()
    if cfg.day_col and cfg.day_col in df_te.columns:
        pred_df[cfg.day_col] = df_te[cfg.day_col].values

    pred_df["y_true"] = y_te_all.values
    pred_df["y_pred"] = y_pred_te
    pred_df["modality"] = modality_name
    pred_path = os.path.join(outdir, f"{modality_name}.predictions.csv")
    pred_df.to_csv(pred_path, index=False)

    # Save metrics
    metrics_df = pd.DataFrame([{"modality": modality_name, **metrics, "n_test": len(pred_df)}])
    metrics_path = os.path.join(outdir, f"{modality_name}.metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    # Save feature selection details
    fs_details_out = fs_details.copy()
    fs_details_out["selected"] = fs_details_out.index.isin(selected_features).astype(int)
    fs_details_out.to_csv(os.path.join(outdir, f"{modality_name}.feature_selection.csv"))

    # Scatter (true vs pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(
        pred_df["y_true"], pred_df["y_pred"],
        s=28, marker="o", facecolors="none", edgecolors=color, linewidths=1.2, alpha=0.9,
    )

    lo = min(pred_df["y_true"].min(), pred_df["y_pred"].min())
    hi = max(pred_df["y_true"].max(), pred_df["y_pred"].max())
    pad = (hi - lo) * 0.05 if hi > lo else 1.0
    lo, hi = lo - pad, hi + pad
    ax.plot([lo, hi], [lo, hi], linewidth=1.0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.set_xlabel("2h iAUC (true)")
    ax.set_ylabel("2h iAUC (pred)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    pr_txt = f"Pearson r={metrics['Pearson_r']:.3f}" if np.isfinite(metrics["Pearson_r"]) else "Pearson r=NA"
    ax.text(0.02, 0.98, pr_txt, transform=ax.transAxes, va="top", ha="left", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{modality_name}.scatter.pdf"))
    plt.close()

    # Calibration plot
    plot_calibration(
        pred_df["y_true"].values,
        pred_df["y_pred"].values,
        out_pdf=os.path.join(outdir, f"{modality_name}.calibration.pdf"),
        title=f"Calibration: {modality_name}",
        n_bins=cfg.calib_bins,
        strategy=cfg.calib_strategy,
        color=color,
    )

    # Day-wise metrics (longitudinal robustness)
    day_metrics = None
    if cfg.day_col and cfg.day_col in pred_df.columns:
        rows = []
        for d, g in pred_df.groupby(cfg.day_col):
            y_t = g["y_true"].values
            y_p = g["y_pred"].values
            if len(g) < 2 or np.std(y_t) == 0 or np.std(y_p) == 0:
                r = np.nan
                p = np.nan
            else:
                r, p = pearsonr(y_t, y_p)
            rows.append({
                "modality": modality_name,
                "day_index": d,
                "n": int(len(g)),
                "Pearson_r": float(r) if np.isfinite(r) else np.nan,
                "Pearson_p": float(p) if np.isfinite(p) else np.nan,
                "MAE": float(mean_absolute_error(y_t, y_p)),
                "RMSE": rmse(y_t, y_p),
            })
        day_metrics = pd.DataFrame(rows).sort_values("day_index")
        day_metrics.to_csv(os.path.join(outdir, f"{modality_name}.metrics_by_day.csv"), index=False)

    return {
        "pred": pred_df,
        "metrics": metrics_df,
        "metrics_by_day": day_metrics if day_metrics is not None else pd.DataFrame(),
        "feature_selection": fs_details_out,
        "selected_features": pd.DataFrame({"feature": selected_features}),
    }


# -----------------------------
# Main CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--labels", required=True, help="CSV with target and metadata (case_id/subject_id/day/ethnicity...).")
    ap.add_argument("--covariates", required=True, help="CSV with baseline covariates (age/sex/BMI/fasting_glucose/meds...).")
    ap.add_argument("--metagenome", required=True, help="CSV of metagenomics features (rows=cases, must include key_col).")
    ap.add_argument("--lipid_metabolome", required=True, help="CSV of lipidome+metabolome features (must include key_col).")
    ap.add_argument("--proteome", required=True, help="CSV of proteomics features (must include key_col).")

    ap.add_argument("--outdir", required=True, help="Output directory.")

    ap.add_argument("--key_col", default="case_id")
    ap.add_argument("--target_col", default="iAUC_2h_true_baseline_sub_pos_dx5")
    ap.add_argument("--group_col", default="subject_id")
    ap.add_argument("--day_col", default="day_index")

    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    # LightGBM selection
    ap.add_argument("--lgb_n_splits", type=int, default=5)
    ap.add_argument("--lgb_n_estimators", type=int, default=2000)
    ap.add_argument("--lgb_top_k", type=int, default=300)

    # Optuna
    ap.add_argument("--optuna_trials", type=int, default=200)

    # Calibration
    ap.add_argument("--calib_bins", type=int, default=10)
    ap.add_argument("--calib_strategy", type=str, default="quantile", choices=["quantile", "uniform"])

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    cfg = TrainConfig(
        key_col=args.key_col,
        target_col=args.target_col,
        group_col=args.group_col,
        day_col=args.day_col if args.day_col else None,        test_size=args.test_size,
        seed=args.seed,
        lgb_n_splits=args.lgb_n_splits,
        lgb_n_estimators=args.lgb_n_estimators,
        lgb_top_k=args.lgb_top_k,
        optuna_trials=args.optuna_trials,
        calib_bins=args.calib_bins,
        calib_strategy=args.calib_strategy,
    )

    os.makedirs(args.outdir, exist_ok=True)

    labels_df = load_table(args.labels)
    covars_df = load_table(args.covariates)
    meta_df   = load_table(args.metagenome)
    lipo_df   = load_table(args.lipid_metabolome)
    prot_df   = load_table(args.proteome)

    # Build "all-omics" by concatenation on key (inner join on key_col)
    def build_allomics(*dfs):
        base = dfs[0][[cfg.key_col]].copy()
        for d in dfs:
            if cfg.key_col not in d.columns:
                raise ValueError(f"Missing {cfg.key_col} in modality features.")
            base = base.merge(d, on=cfg.key_col, how="inner")
        # remove duplicate key_col columns after merges
        # (merge keeps one key col)
        return base

    allomics_df = build_allomics(meta_df, lipo_df, prot_df)

    modalities = {
        "metagenome": (meta_df,  "#2C74B6"),
        "lipid_metabolome": (lipo_df, "#BD7A99"),
        "proteome": (prot_df, "#6D9C56"),
        "all_omics": (allomics_df, "#CE8E4C"),
    }

    metrics_all = []
    for name, (feat_df, color) in modalities.items():
        out = run_one_modality(
            modality_name=name,
            labels_df=labels_df,
            covars_df=covars_df,
            feats_df=feat_df,
            outdir=args.outdir,
            cfg=cfg,
            color=color,
        )
        metrics_all.append(out["metrics"])

    metrics_all = pd.concat(metrics_all, axis=0, ignore_index=True)
    metrics_all.to_csv(os.path.join(args.outdir, "ALL_OMICS_MODELS.metrics.csv"), index=False)

    print("Done. Saved omics-model metrics to:", os.path.join(args.outdir, "ALL_OMICS_MODELS.metrics.csv"))


if __name__ == "__main__":
    main()
