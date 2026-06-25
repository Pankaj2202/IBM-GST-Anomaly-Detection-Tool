"""
GST Anomaly Detection Tool - Deterministic Rule Engine
JK Cement | Supply Chain Accounts Payable

Implements the auditable validation layer. Every flag here is explainable and
maps directly to a documented business rule, so a reviewer can see *why* an
invoice was raised. The ML layer (gst_model.py) sits on top of these signals.

Validation algorithms (from problem statement) + GST checkpoints (Pawan's mail):
  R1  Invoice number present & vendor in active vendor list
  R2  Invoice date not in the future (vs posting date)
  R3  Invoice amount deviates from vendor's historical median (mean/median %)
  R4  Tax code valid & consistent with Inter/Intra (Ship-To vs Bill-To state)
  R5  Business place valid & maps to a known JK recipient GSTIN
  R6  (checkpoint) Old / stale invoice (filing delay)
  R7  (checkpoint) Round-amount / one-time-vendor soft signals
"""

import numpy as np
import pandas as pd
from gst_config import THRESHOLDS

CANON_COLS = [
    "co_code", "vendor_code", "vendor_name", "invoice_no", "doc_date",
    "doc_amount", "pstng_date", "document_no", "doc_type", "posting_key",
    "text", "business_place", "tax_code", "vendor_gstin",
]

# maps many possible source headers -> canonical name
_COL_ALIASES = {
    "cocd": "co_code", "co_code": "co_code",
    "account": "vendor_code", "vendor": "vendor_code", "vendor_code": "vendor_code",
    "vendorname": "vendor_name", "vendor_name": "vendor_name", "suppliername(pr)": "vendor_name",
    "reference": "invoice_no", "invoice": "invoice_no", "invoice_no": "invoice_no",
    "documentnumber(pr)": "invoice_no",
    "docdate": "doc_date", "doc_date": "doc_date", "documentdate(pr)": "doc_date",
    "docamt": "doc_amount", "locamt": "doc_amount", "doc_amount": "doc_amount",
    "invoicevalue(pr)": "doc_amount",
    "pstngdate": "pstng_date", "pstng_date": "pstng_date", "glpostingdate": "pstng_date",
    "documentno": "document_no", "document_no": "document_no",
    "accountingvouchernumber": "document_no",
    "type": "doc_type", "doctype(pr)": "doc_type", "doc_type": "doc_type",
    "pk": "posting_key", "posting_key": "posting_key",
    "text": "text",
    "plantcode": "business_place", "business_place": "business_place",
    "businessplace": "business_place",
    "tax_code": "tax_code", "taxcode": "tax_code",
    "suppliergstin(pr)": "vendor_gstin", "vendor_gstin": "vendor_gstin",
    "vendorgstin": "vendor_gstin",
}


def standardize(df):
    """Coerce an arbitrary source frame into the canonical schema."""
    df = df.copy()
    rename = {}
    for c in df.columns:
        key = str(c).strip().lower().replace(" ", "")
        if key in _COL_ALIASES:
            rename[c] = _COL_ALIASES[key]
    df = df.rename(columns=rename)
    # when several source cols map to the same canonical name, keep the first
    df = df.loc[:, ~df.columns.duplicated()]
    for c in CANON_COLS:
        if c not in df.columns:
            df[c] = np.nan
    out = df[CANON_COLS].copy()
    # type coercion
    out["doc_date"] = _to_datetime(out["doc_date"])
    out["pstng_date"] = _to_datetime(out["pstng_date"])
    out["doc_amount"] = pd.to_numeric(out["doc_amount"], errors="coerce")
    out["document_no"] = pd.to_numeric(out["document_no"], errors="coerce")
    for c in ["vendor_name", "invoice_no", "business_place", "tax_code",
              "vendor_gstin", "doc_type", "vendor_code"]:
        out[c] = out[c].astype(str).str.strip()
        out.loc[out[c].isin(["nan", "None", ""]), c] = np.nan
    out["business_place"] = out["business_place"].astype(str).str.replace(r"\.0$", "", regex=True)
    out["tax_code"] = out["tax_code"].str.upper()
    out["vendor_gstin"] = out["vendor_gstin"].str.upper()
    out["vendor_name_norm"] = (out["vendor_name"].fillna("").str.upper()
                               .str.replace(r"[^A-Z0-9]", "", regex=True))
    return out


def _to_datetime(s):
    """Handle both real datetimes and Excel serial numbers in one column."""
    s = s.copy()
    dt = pd.to_datetime(s, errors="coerce")
    # rows that came as excel serials (numbers ~40000-60000)
    num = pd.to_numeric(s, errors="coerce")
    serial_mask = dt.isna() & num.between(20000, 60000)
    if serial_mask.any():
        dt.loc[serial_mask] = pd.to_datetime(
            num[serial_mask], unit="D", origin="1899-12-30")
    return dt


# ---------------------------------------------------------------------------
# Rule engine
# ---------------------------------------------------------------------------
def run_rules(df, ref, today=None):
    """
    Apply all validation rules. Returns the input frame plus:
      flag_* boolean columns, rule_flag_count, rule_reasons (text).
    `ref` is the dict from build_reference_pack(); `df` must be standardized.
    """
    df = df.copy()
    today = pd.Timestamp(today) if today else pd.Timestamp.today().normalize()

    # vendor historical median (for amount deviation) computed within this batch.
    # Fall back to vendor_name / sentinel so an all-missing vendor_code never
    # collapses the groupby (which would break the amount-deviation maths).
    grp_key = (df["vendor_code"].fillna(df["vendor_name"])
               .fillna("UNKNOWN").astype(str))
    vmed = df.groupby(grp_key)["doc_amount"].transform(lambda x: x.abs().median())
    vcount = df.groupby(grp_key)["doc_amount"].transform("count")

    filing_delay = (df["pstng_date"] - df["doc_date"]).dt.days

    # ---- R2  Invoice date in the future --------------------------------
    df["flag_future_date"] = (df["doc_date"] > df["pstng_date"]) | (df["doc_date"] > today)

    # ---- R6  Old / stale invoice ---------------------------------------
    df["flag_old_invoice"] = filing_delay > THRESHOLDS["old_invoice_days"]

    # ---- R1  Invoice number missing / vendor not in active list --------
    df["flag_invoice_no_missing"] = df["invoice_no"].isna() | (df["invoice_no"] == "")
    in_list_name = df["vendor_name_norm"].isin(ref["active_vendor_names"])
    in_list_gstin = df["vendor_gstin"].isin(ref["active_vendor_gstins"])
    has_gstin = df["vendor_gstin"].notna()
    df["flag_vendor_not_in_master"] = ~(in_list_name | (has_gstin & in_list_gstin))
    # one-time vendors are expected exceptions, not master-list failures
    onetime = df["vendor_name"].fillna("").str.upper().str.contains("ONE TIME")
    df.loc[onetime, "flag_vendor_not_in_master"] = False
    df["flag_one_time_vendor"] = onetime

    # ---- R3  Amount deviation from vendor median -----------------------
    dev_pct = np.where(vmed > 0, (df["doc_amount"].abs() - vmed).abs() / vmed * 100, 0)
    df["amount_dev_pct"] = np.round(dev_pct, 1)
    df["flag_amount_outlier"] = (
        (vcount >= THRESHOLDS["min_vendor_history"]) &
        (dev_pct > THRESHOLDS["amount_dev_pct"])
    )

    # ---- R5  Business place validity -----------------------------------
    bp_known = df["business_place"].isin(ref["valid_business_places"])
    df["flag_business_place_invalid"] = df["business_place"].notna() & ~bp_known

    # ---- R4  Tax code validity -----------------------------------------
    tc_known = df["tax_code"].isin(ref["valid_tax_codes"])
    df["flag_tax_code_invalid"] = df["tax_code"].notna() & ~tc_known

    # ---- R7  Round-amount soft signal ----------------------------------
    amt = df["doc_amount"].abs().fillna(0)
    df["flag_round_amount"] = (amt >= 100000) & (amt % THRESHOLDS["round_amount_modulo"] == 0)

    # ---- aggregate ------------------------------------------------------
    flag_cols = [c for c in df.columns if c.startswith("flag_")
                 and c != "flag_one_time_vendor"]
    df["rule_flag_count"] = df[flag_cols].sum(axis=1)

    reason_map = {
        "flag_future_date": "Invoice date in future",
        "flag_old_invoice": "Stale invoice (filing delay)",
        "flag_invoice_no_missing": "Invoice number missing",
        "flag_vendor_not_in_master": "Vendor not in active master",
        "flag_amount_outlier": "Amount deviates from vendor norm",
        "flag_business_place_invalid": "Unknown business place",
        "flag_tax_code_invalid": "Invalid tax code",
        "flag_round_amount": "Large round-number amount",
    }

    def reasons(row):
        return "; ".join(reason_map[c] for c in flag_cols if row[c])
    df["rule_reasons"] = df.apply(reasons, axis=1)
    df["filing_delay_days"] = filing_delay
    return df
