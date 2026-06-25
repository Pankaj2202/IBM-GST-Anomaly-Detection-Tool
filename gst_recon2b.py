"""
GST Anomaly Detection Tool - GSTR-2B vs Purchase Register Reconciliation
JK Cement | Supply Chain Accounts Payable

This is the fake/unsupported-ITC detector. It is DETERMINISTIC and auditable
(not ML): for every 2B-vs-PR record it classifies the mismatch type, computes
the ITC-at-risk (PR-claimed ITC not backed by the supplier's 2B filing), and
assigns a risk band. The headline number for leadership is total ITC-at-risk.

Input: the GSTR-2B vs PR mismatch extract (columns like IGST(2B)/IGST(PR),
TaxableValue(2B)/(PR), InvoiceValue(2B)/(PR), EligibilityIndicator,
ReverseChargeFlag(2B)/(PR), SupplierGSTIN(2B)/(PR), DocumentDate(2B)/(PR)).
"""

import numpy as np
import pandas as pd

ROUNDING_TOLERANCE = 1.0      # |2B-PR| <= this rupee value treated as a rounding match


def _num(df, col):
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def _txt(df, col):
    if col not in df.columns:
        return pd.Series("", index=df.index)
    s = df[col].where(df[col].notna(), "")
    s = (s.astype(str).str.strip()
         .str.replace("`", "", regex=False))            # strip Excel text-guard backticks
    return s.where(~s.str.lower().isin(["nan", "none", "nat", ""]), "")


def reconcile(df):
    """Classify each 2B-vs-PR row and compute ITC-at-risk + risk band."""
    out = pd.DataFrame(index=df.index)

    # --- identity / display fields (backticks stripped) ----------------
    out["Supplier GSTIN (PR)"] = _txt(df, "SupplierGSTIN(PR)")
    out["Supplier GSTIN (2B)"] = _txt(df, "SupplierGSTIN(2B)")
    name = _txt(df, "SupplierName(PR)")
    out["Supplier Name"] = name.where(name != "", _txt(df, "SupplierName(2B)"))
    out["Invoice No (PR)"] = _txt(df, "DocumentNumber(PR)")
    out["Inv Date (PR)"] = pd.to_datetime(df.get("DocumentDate(PR)"), errors="coerce")
    out["Inv Date (2B)"] = pd.to_datetime(df.get("DocumentDate(2B)"), errors="coerce")
    out["Eligibility"] = _txt(df, "EligibilityIndicator")
    out["Category"] = _txt(df, "Category")

    # --- ITC claimed (PR) vs supported (2B) ----------------------------
    itc_pr = _num(df, "IGST(PR)") + _num(df, "CGST(PR)") + _num(df, "SGST(PR)")
    itc_2b = _num(df, "IGST(2B)") + _num(df, "CGST(2B)") + _num(df, "SGST(2B)")
    out["ITC Claimed (PR)"] = itc_pr.round(2)
    out["ITC in 2B"] = itc_2b.round(2)

    tax_pr = _num(df, "TotalTax(PR)"); tax_2b = _num(df, "TotalTax(2B)")
    tv_pr = _num(df, "TaxableValue(PR)"); tv_2b = _num(df, "TaxableValue(2B)")
    iv_pr = _num(df, "InvoiceValue(PR)"); iv_2b = _num(df, "InvoiceValue(2B)")
    rc_pr = _txt(df, "ReverseChargeFlag(PR)").str.upper()
    rc_2b = _txt(df, "ReverseChargeFlag(2B)").str.upper()

    tol = ROUNDING_TOLERANCE
    # --- mismatch type (first matching rule wins, severity order) ------
    cond_2b_missing = (itc_2b <= tol) & (itc_pr > tol)
    cond_ineligible = (out["Eligibility"] == "NO") & (itc_pr > tol)
    cond_gstin = (out["Supplier GSTIN (2B)"] != "") & \
                 (out["Supplier GSTIN (PR)"] != out["Supplier GSTIN (2B)"])
    cond_rcm = (rc_2b != "") & (rc_pr != rc_2b)
    cond_tax = (tax_2b - tax_pr).abs() > tol
    cond_tv = (tv_2b - tv_pr).abs() > tol
    cond_iv = (iv_2b - iv_pr).abs() > tol
    cond_date = out["Inv Date (2B)"].notna() & out["Inv Date (PR)"].notna() & \
                (out["Inv Date (2B)"] != out["Inv Date (PR)"])

    # ITC at risk: unsupported portion normally; full claim when ineligible
    at_risk = (itc_pr - itc_2b).clip(lower=0)
    at_risk = np.where(cond_ineligible, itc_pr, at_risk)
    out["ITC at Risk"] = np.round(at_risk, 2)

    mismatch = np.select(
        [cond_2b_missing, cond_ineligible, cond_gstin, cond_rcm,
         cond_tax, cond_tv, cond_iv, cond_date],
        ["2B-Missing: ITC claimed not in supplier 2B",
         "Ineligible ITC claimed",
         "Supplier GSTIN mismatch",
         "Reverse-charge (RCM/FCM) treatment mismatch",
         "Tax amount mismatch",
         "Taxable value mismatch",
         "Invoice value mismatch",
         "Document date mismatch"],
        default="Matched / rounding only")
    out["Mismatch Type"] = mismatch

    # --- risk band ------------------------------------------------------
    high = cond_2b_missing | cond_ineligible | cond_gstin
    med = ~high & (cond_rcm | cond_tax | ((out["ITC at Risk"] > tol)))
    band = np.where(high, "HIGH", np.where(med, "MEDIUM", "LOW"))
    # any row with no real ITC at risk and only value/date diff -> LOW
    out["Risk Band"] = band
    out["Doc Type (PR)"] = _txt(df, "DocType(PR)")
    out["Plant"] = _txt(df, "PlantCode")
    return out


def summarize(recon):
    """Return summary tables for the dashboard."""
    by_type = (recon.groupby("Mismatch Type")
               .agg(Invoices=("ITC at Risk", "size"),
                    ITC_at_Risk=("ITC at Risk", "sum"))
               .sort_values("ITC_at_Risk", ascending=False).reset_index())
    by_band = (recon.groupby("Risk Band")
               .agg(Invoices=("ITC at Risk", "size"),
                    ITC_at_Risk=("ITC at Risk", "sum")).reset_index())
    totals = {
        "rows": len(recon),
        "total_itc_at_risk": float(recon["ITC at Risk"].sum()),
        "high_rows": int((recon["Risk Band"] == "HIGH").sum()),
        "high_itc": float(recon.loc[recon["Risk Band"] == "HIGH", "ITC at Risk"].sum()),
    }
    return by_type, by_band, totals
