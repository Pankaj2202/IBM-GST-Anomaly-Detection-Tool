"""
GST Anomaly Detection Tool - Machine Learning Layer
JK Cement | Supply Chain Accounts Payable

Random Forest classifier that converts the engineered invoice signals into a
single calibrated fraud-RISK probability and ranks invoices for human review.

Target = 1 if an invoice was historically flagged as an ARM anomaly
(DollarValueDiff / FutureInvoice / InvoicePatternMismatch / OldInvoice),
else 0. This is anomaly/exception risk - NOT a confirmed-fraud verdict.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, precision_recall_fscore_support)

FEATURES = [
    "log_amount", "filing_delay_days", "is_future_invoice", "is_old_invoice",
    "amount_dev_pct", "vendor_txn_count", "log_vendor_turnover",
    "vendor_not_in_master", "ref_len", "ref_digit_ratio", "ref_is_numeric",
    "ref_has_special", "is_one_time_vendor", "posting_month",
    "doc_type_code", "round_amount",
]


def engineer_features(df, ref):
    """Build the model feature matrix from a standardized + rule-run frame."""
    df = df.copy()
    amt = df["doc_amount"].abs().fillna(0)
    df["log_amount"] = np.log1p(amt)

    delay = df.get("filing_delay_days")
    if delay is None:
        delay = (df["pstng_date"] - df["doc_date"]).dt.days
    df["filing_delay_days"] = pd.to_numeric(delay, errors="coerce").fillna(0)
    df["is_future_invoice"] = (df["doc_date"] > df["pstng_date"]).astype(int)
    df["is_old_invoice"] = (df["filing_delay_days"] > 180).astype(int)

    # vendor-level aggregates (turnover proxy + history depth) — robust to a
    # missing vendor_code by falling back to vendor_name / sentinel.
    grp_key = (df["vendor_code"].fillna(df["vendor_name"])
               .fillna("UNKNOWN").astype(str))
    g = df.groupby(grp_key)["doc_amount"]
    df["vendor_txn_count"] = g.transform("count")
    df["vendor_turnover"] = g.transform(lambda x: x.abs().sum())
    df["log_vendor_turnover"] = np.log1p(df["vendor_turnover"])
    vmed = g.transform(lambda x: x.abs().median())
    df["amount_dev_pct"] = np.where(vmed > 0, (amt - vmed).abs() / vmed * 100, 0)

    # active-vendor-master membership
    if "flag_vendor_not_in_master" in df.columns:
        df["vendor_not_in_master"] = df["flag_vendor_not_in_master"].astype(int)
    else:
        in_name = df["vendor_name_norm"].isin(ref["active_vendor_names"])
        df["vendor_not_in_master"] = (~in_name).astype(int)

    # invoice-number pattern features (drives pattern-mismatch detection)
    ref_str = df["invoice_no"].fillna("").astype(str)
    df["ref_len"] = ref_str.str.len()
    digit_cnt = ref_str.str.count(r"\d")
    df["ref_digit_ratio"] = np.where(df["ref_len"] > 0, digit_cnt / df["ref_len"], 0)
    df["ref_is_numeric"] = ref_str.str.match(r"^\d+$").fillna(False).astype(int)
    df["ref_has_special"] = ref_str.str.contains(r"[^A-Za-z0-9/\-]").fillna(False).astype(int)

    df["is_one_time_vendor"] = (df["vendor_name"].fillna("").str.upper()
                                .str.contains("ONE TIME").astype(int))
    df["posting_month"] = df["pstng_date"].dt.month.fillna(0).astype(int)
    df["doc_type_code"] = df["doc_type"].fillna("NA").astype("category").cat.codes
    df["round_amount"] = ((amt >= 100000) & (amt % 10000 == 0)).astype(int)
    return df


def train_model(df_feat, label_col="is_anomaly", random_state=42):
    """Train + evaluate the Random Forest. Returns (model, metrics dict)."""
    X = df_feat[FEATURES].fillna(0)
    y = df_feat[label_col].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=random_state)

    model = RandomForestClassifier(
        n_estimators=300, max_depth=18, min_samples_leaf=5,
        class_weight="balanced", n_jobs=-1, random_state=random_state)
    model.fit(X_tr, y_tr)

    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_te, pred, average="binary", zero_division=0)

    metrics = {
        "n_train": len(X_tr), "n_test": len(X_te),
        "positives_total": int(y.sum()), "positive_rate": float(y.mean()),
        "accuracy": float((pred == y_te).mean()),
        "precision": float(p), "recall": float(r), "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_te, proba)),
        "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
        "report": classification_report(y_te, pred, zero_division=0,
                                         target_names=["Normal", "Anomaly"]),
        "feature_importance": sorted(
            zip(FEATURES, model.feature_importances_),
            key=lambda t: t[1], reverse=True),
    }
    return model, metrics


def score(df_feat, model):
    """Attach model risk probability + risk band to an engineered frame."""
    from gst_config import THRESHOLDS
    df = df_feat.copy()
    df["risk_score"] = model.predict_proba(df[FEATURES].fillna(0))[:, 1]
    bins = [-0.01, THRESHOLDS["med_risk_score"],
            THRESHOLDS["high_risk_score"], 1.01]
    df["risk_band"] = pd.cut(df["risk_score"], bins=bins,
                             labels=["LOW", "MEDIUM", "HIGH"])
    return df
