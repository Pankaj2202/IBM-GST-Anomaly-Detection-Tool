"""
GST Anomaly Detection Tool - Advanced Forensic Analytics
JK Cement | Supply Chain Accounts Payable

Market-grade techniques layered on top of the rule + supervised-ML engines,
grounded in how DGARM/DGGI and forensic auditors actually detect fake-ITC and
shell-company fraud:

  1. Benford's Law        - first-digit test for fabricated amounts (forensic
                            accounting standard; Nigrini MAD conformity).
  2. Isolation Forest     - UNSUPERVISED anomaly detection; finds novel outliers
                            the supervised model was never trained on.
  3. Vendor-network signals - shell-company / circular-trading footprint:
                            one PAN -> many GSTINs, one GSTIN -> many names,
                            invalid/structurally-bad GSTINs concentrating ITC.
  4. Duplicate / split    - same invoice posted twice; split invoicing (same
                            vendor, near-identical amount, close dates).
  5. Round / threshold    - amounts engineered to round figures or just under
                            approval thresholds.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from gstin_validator import validate as gstin_validate


# ---------------------------------------------------------------------------
# 1. Benford's Law
# ---------------------------------------------------------------------------
BENFORD_EXP = {d: np.log10(1 + 1 / d) for d in range(1, 10)}


def benford_first_digit(amounts):
    a = pd.to_numeric(pd.Series(amounts), errors="coerce").dropna().abs()
    a = a[a >= 1]
    fd = (a.astype("int64").astype(str).str.lstrip("0").str[0])
    fd = pd.to_numeric(fd, errors="coerce").dropna()
    fd = fd[(fd >= 1) & (fd <= 9)].astype(int)
    obs = fd.value_counts(normalize=True).reindex(range(1, 10), fill_value=0).sort_index()
    exp = pd.Series(BENFORD_EXP).sort_index()
    mad = float(np.abs(obs.values - exp.values).mean())
    verdict = ("Close conformity" if mad < 0.006 else
               "Acceptable conformity" if mad < 0.012 else
               "Marginal — watch" if mad < 0.015 else
               "Nonconformity — investigate")
    table = pd.DataFrame({"digit": list(range(1, 10)),
                          "observed": obs.values, "expected": exp.values,
                          "count": fd.value_counts().reindex(range(1, 10), fill_value=0).sort_index().values})
    return table, mad, verdict


# ---------------------------------------------------------------------------
# 2. Isolation Forest (unsupervised)
# ---------------------------------------------------------------------------
def isolation_forest(feat_df, features, contamination=0.05):
    X = feat_df[[f for f in features if f in feat_df.columns]].fillna(0)
    if len(X) < 20:
        return pd.Series(0.0, index=feat_df.index)
    m = IsolationForest(n_estimators=200, contamination=contamination,
                        random_state=42, n_jobs=-1)
    m.fit(X)
    raw = -m.score_samples(X)          # higher = more anomalous
    # min-max to 0-1 for readability
    lo, hi = raw.min(), raw.max()
    norm = (raw - lo) / (hi - lo) if hi > lo else raw * 0
    return pd.Series(norm, index=feat_df.index)


# ---------------------------------------------------------------------------
# 3. Vendor-network / shell-company signals
# ---------------------------------------------------------------------------
def vendor_network(df, gstin_col, name_col, amount_col=None):
    g = df[gstin_col].astype(str).str.strip().str.upper().str.replace("`", "", regex=False)
    nm = df[name_col].astype(str).str.strip().str.upper()
    pan = g.str[2:12]
    work = pd.DataFrame({"gstin": g, "name": nm, "pan": pan})

    pan_gstins = work.groupby("pan")["gstin"].nunique()
    gstin_names = work.groupby("gstin")["name"].nunique()
    name_gstins = work.groupby("name")["gstin"].nunique()

    work["pan_multi_gstin"] = work["pan"].map(pan_gstins).fillna(0) > 1
    work["gstin_multi_name"] = work["gstin"].map(gstin_names).fillna(0) > 1
    work["name_multi_gstin"] = work["name"].map(name_gstins).fillna(0) > 1
    # structural GSTIN validity (cached over unique values)
    uniq = {x: gstin_validate(x)["valid"] for x in work["gstin"].dropna().unique()}
    work["gstin_invalid"] = ~work["gstin"].map(uniq).fillna(False)

    flags = pd.DataFrame(index=df.index)
    flags["pan_multi_gstin"] = work["pan_multi_gstin"].values
    flags["gstin_multi_name"] = work["gstin_multi_name"].values
    flags["name_multi_gstin"] = work["name_multi_gstin"].values
    flags["gstin_invalid"] = work["gstin_invalid"].values

    # network risk score (weighted) — shell/circular footprint
    flags["network_risk"] = (
        0.40 * flags["gstin_invalid"] +
        0.25 * flags["gstin_multi_name"] +
        0.20 * flags["pan_multi_gstin"] +
        0.15 * flags["name_multi_gstin"]).round(3)

    # suspicious clusters for display
    clusters = pd.DataFrame({
        "GSTINs sharing a PAN": [int((pan_gstins > 1).sum())],
        "GSTINs with multiple names": [int((gstin_names > 1).sum())],
        "Names with multiple GSTINs": [int((name_gstins > 1).sum())],
        "Structurally invalid GSTINs": [int(sum(1 for v in uniq.values() if not v))],
    })
    return flags, clusters


# ---------------------------------------------------------------------------
# 4. Duplicate / split-invoice detection
# ---------------------------------------------------------------------------
def duplicate_invoices(df, vendor_col, inv_col, amount_col, date_col,
                       amount_tol=1.0, day_window=7):
    d = pd.DataFrame({
        "vendor": df[vendor_col].astype(str).str.strip().str.upper(),
        "inv": df[inv_col].astype(str).str.strip().str.upper().str.replace("`", "", regex=False),
        "amt": pd.to_numeric(df[amount_col], errors="coerce").abs(),
        "date": pd.to_datetime(df[date_col], errors="coerce"),
    })
    flags = pd.DataFrame(index=df.index)
    # exact duplicate: same vendor + invoice number appearing more than once
    key = d["vendor"] + "|" + d["inv"]
    flags["exact_duplicate"] = (key.map(key.value_counts()) > 1) & (d["inv"] != "")
    # split-invoice: same vendor + same amount within a short window, diff invoice
    d_sorted = d.assign(_i=range(len(d))).sort_values(["vendor", "amt", "date"])
    split = np.zeros(len(d), dtype=bool)
    prev = {}
    for vendor, amt, date, idx, inv in zip(d_sorted["vendor"], d_sorted["amt"],
                                           d_sorted["date"], d_sorted["_i"], d_sorted["inv"]):
        pkey = (vendor, round(amt, 0) if pd.notna(amt) else None)
        if pkey in prev and pd.notna(date) and pd.notna(prev[pkey][0]):
            if abs((date - prev[pkey][0]).days) <= day_window and inv != prev[pkey][1]:
                split[idx] = True
                split[prev[pkey][2]] = True
        prev[pkey] = (date, inv, idx)
    flags["split_invoice"] = split
    return flags


# ---------------------------------------------------------------------------
# 5. Round-number / threshold gaming
# ---------------------------------------------------------------------------
def round_threshold(amounts, thresholds=(50000, 100000, 200000, 500000), band=0.03):
    a = pd.to_numeric(pd.Series(amounts), errors="coerce").abs()
    flags = pd.DataFrame(index=a.index)
    flags["round_number"] = (a >= 50000) & (a % 10000 == 0)
    just_under = np.zeros(len(a), dtype=bool)
    for t in thresholds:
        just_under |= a.between(t * (1 - band), t - 0.01)
    flags["just_under_threshold"] = just_under
    return flags
