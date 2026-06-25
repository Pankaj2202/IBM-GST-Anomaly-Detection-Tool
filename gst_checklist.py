"""
GST Anomaly Detection Tool - 12-Point Process Checklist
JK Cement | Supply Chain Accounts Payable

Runs JK Cement's process-specific GST checklist against the GSTR-2B vs PR
extract (the field-richest source: both GSTINs, state, tax break-up,
eligibility, RCM flags, invoice no/date/value).

Each invoice gets PASS / FAIL / NA per check, an overall status, and the list
of failed checks. Two checks need inputs not present in the data (the physical
invoice copy / PO) and are returned as NA with a note — see CHECK_NOTES.
"""

import numpy as np
import pandas as pd
from gstin_validator import validate, JK_PAN

ROUND_TOL = 1.0

CHECKS = {
    "C1_SingleDocPerInvoice": "Single SAP doc per invoice (FLYASH exempt)",
    "C2_CompleteInvoice": "Invoice complete (no mandatory field blank)",
    "C3_TitleAndName": "Titled TAX INVOICE/CREDIT NOTE, in JK Cement name",
    "C4_InvoiceDate": "Correct invoice date (not future / not blank)",
    "C5_InvoiceNumber": "Correct invoice number (present, 2B=PR)",
    "C6_InvoiceAmount": "Correct amount (value = taxable + tax)",
    "C7_BusinessPlace": "Correct business place / recipient GSTIN (JK)",
    "C8_SupplierGSTIN": "Correct supplier GSTIN (valid, 2B=PR)",
    "C9_InterIntraTax": "Correct Inter/Intra tax vs supplier-recipient state",
    "C10_Eligibility": "Correct eligible/ineligible ITC treatment",
    "C11_RCM_FCM": "Correct reverse/forward charge treatment",
    "C12_Description": "Correct invoice description (PO / invoice copy)",
    "C13_DocumentType": "Valid SAP document type, charge-consistent (SOP §4)",
    "C14_GSTRate": "GST rate valid (tax ÷ taxable), Inter/Intra correct (ZGSTR2N)",
    "C15_RecipientState": "Recipient GSTIN state matches business place (ZGSTR2N)",
    "C16_HSN": "HSN present (as per vendor invoice — GSTIN checklist §2.2)",
}
# checks that need data not in the extract
CHECK_NOTES = {
    "C3_TitleAndName": "Title needs the invoice copy (OCR). Only the 'JK Cement "
                       "name' part is auto-checked via recipient PAN.",
    "C12_Description": "Needs PO master (PO items) or the invoice copy (OCR).",
    "C16_HSN": "Needs the HSN field from the vendor invoice (not in this extract).",
}


def _num(df, c):
    return pd.to_numeric(df.get(c), errors="coerce").fillna(0.0) if c in df.columns \
        else pd.Series(0.0, index=df.index)


def _txt(df, c):
    if c not in df.columns:
        return pd.Series("", index=df.index)
    s = df[c].where(df[c].notna(), "")
    s = s.astype(str).str.strip().str.replace("`", "", regex=False)
    return s.where(~s.str.lower().isin(["nan", "none", "nat", ""]), "")


def run_checklist(df):
    n = len(df)
    R = pd.DataFrame(index=df.index)

    sup_pr = _txt(df, "SupplierGSTIN(PR)").str.upper()
    sup_2b = _txt(df, "SupplierGSTIN(2B)").str.upper()
    rec_pr = _txt(df, "RecipientGSTIN(PR)").str.upper()
    rec_2b = _txt(df, "RecipientGSTIN(2B)").str.upper()
    name = _txt(df, "SupplierName(PR)")
    invno_pr = _txt(df, "DocumentNumber(PR)")
    invno_2b = _txt(df, "DocumentNumber(2B)")
    voucher = _txt(df, "AccountingVoucherNumber")
    date_pr = pd.to_datetime(df.get("DocumentDate(PR)"), errors="coerce")
    date_2b = pd.to_datetime(df.get("DocumentDate(2B)"), errors="coerce")
    post = pd.to_datetime(df.get("GLPostingDate"), errors="coerce")

    R["Supplier"] = name
    R["Invoice No"] = invno_pr
    R["Supplier GSTIN"] = sup_pr

    # pre-compute GSTIN validations (unique to save time)
    uniq = pd.unique(pd.concat([sup_pr, rec_pr]).dropna())
    vmap = {g: validate(g) for g in uniq if g}

    def vget(g, key):
        return vmap.get(g, {}).get(key, "")

    # --- C1 single SAP doc per invoice (FLYASH exempt) ------------------
    key = sup_pr + "|" + invno_pr
    docs_per_inv = key.groupby(key).transform("size")  # rows sharing the same invoice
    distinct_vouchers = (df.assign(_k=key, _v=voucher)
                         .groupby("_k")["_v"].transform("nunique"))
    is_flyash = name.str.upper().str.contains("FLYASH|FLY ASH|FLY-ASH", regex=True)
    R["C1_SingleDocPerInvoice"] = np.where(
        is_flyash, "NA",
        np.where((invno_pr != "") & (distinct_vouchers > 1), "FAIL", "PASS"))

    # --- C2 complete invoice -------------------------------------------
    complete = (invno_pr != "") & date_pr.notna() & (sup_pr != "") & \
               (rec_pr != "") & (_num(df, "InvoiceValue(PR)") > 0)
    R["C2_CompleteInvoice"] = np.where(complete, "PASS", "FAIL")

    # --- C3 title + JK name (only name part auto-checkable) ------------
    rec_is_jk = rec_pr.str[2:12] == JK_PAN
    R["C3_TitleAndName"] = np.where(rec_is_jk, "PASS(name only)", "FAIL")

    # --- C4 invoice date ------------------------------------------------
    future = date_pr.notna() & post.notna() & (date_pr > post)
    R["C4_InvoiceDate"] = np.where(date_pr.isna(), "FAIL",
                                   np.where(future, "FAIL", "PASS"))

    # --- C5 invoice number (present + 2B==PR) --------------------------
    num_ok = (invno_pr != "") & ((invno_2b == "") | (invno_2b == invno_pr))
    R["C5_InvoiceNumber"] = np.where(num_ok, "PASS", "FAIL")

    # --- C6 amount correct: PR invoice value matches supplier 2B -------
    val_pr = _num(df, "InvoiceValue(PR)")
    val_2b = _num(df, "InvoiceValue(2B)")
    has_2b = (_txt(df, "SupplierGSTIN(2B)") != "") | (val_2b > 0)
    R["C6_InvoiceAmount"] = np.where(~has_2b, "NA",
                                     np.where((val_2b - val_pr).abs() <= 2.0,
                                              "PASS", "FAIL"))

    # --- C7 business place / recipient GSTIN ---------------------------
    rec_valid = rec_pr.map(lambda g: vget(g, "valid") is True)
    rec_match = (rec_2b == "") | (rec_2b == rec_pr)
    R["C7_BusinessPlace"] = np.where(rec_valid & rec_is_jk & rec_match, "PASS", "FAIL")

    # --- C8 supplier GSTIN ---------------------------------------------
    sup_valid = sup_pr.map(lambda g: vget(g, "valid") is True)
    sup_match = (sup_2b == "") | (sup_2b == sup_pr)
    R["C8_SupplierGSTIN"] = np.where(sup_valid & sup_match, "PASS", "FAIL")

    # --- C9 Inter/Intra tax correctness --------------------------------
    sup_state = sup_pr.str[:2]
    rec_state = rec_pr.str[:2]
    igst = _num(df, "IGST(PR)"); cgst = _num(df, "CGST(PR)"); sgst = _num(df, "SGST(PR)")
    intra = sup_state == rec_state
    # Intra => CGST+SGST present, IGST 0 ; Inter => IGST present, CGST/SGST 0
    intra_ok = intra & (igst <= ROUND_TOL) & ((cgst + sgst) > ROUND_TOL)
    inter_ok = (~intra) & (igst > ROUND_TOL) & ((cgst + sgst) <= ROUND_TOL)
    no_tax = (igst + cgst + sgst) <= ROUND_TOL  # nil/exempt -> not a tax-type error
    valid_states = (sup_state.str.len() == 2) & (rec_state.str.len() == 2)
    R["C9_InterIntraTax"] = np.where(
        ~valid_states | no_tax, "NA",
        np.where(intra_ok | inter_ok, "PASS", "FAIL"))

    # --- C10 eligibility ------------------------------------------------
    elig = _txt(df, "EligibilityIndicator").str.upper()
    itc = igst + cgst + sgst
    R["C10_Eligibility"] = np.where(elig == "", "NA",
                                    np.where((elig == "NO") & (itc > ROUND_TOL),
                                             "FAIL", "PASS"))

    # --- C11 RCM / FCM --------------------------------------------------
    rc_pr = _txt(df, "ReverseChargeFlag(PR)").str.upper()
    rc_2b = _txt(df, "ReverseChargeFlag(2B)").str.upper()
    rcm_match = (rc_2b == "") | (rc_pr == rc_2b)
    R["C11_RCM_FCM"] = np.where(rc_pr == "", "NA",
                                np.where(rcm_match, "PASS", "FAIL"))

    # --- C12 description (needs PO / invoice copy) ---------------------
    R["C12_Description"] = "NA"

    # --- C13/C14/C15 process-specific checks (SOP §4 + ZGSTR2N) --------
    from gst_process import run_process_checks
    proc = run_process_checks(df)
    R["C13_DocumentType"] = np.where(proc["doc_type"].values == "", "NA",
                                     np.where(proc["doc_ok"].values, "PASS", "FAIL"))
    R["C14_GSTRate"] = np.where(proc["rate_ok"].values, "PASS", "FAIL")
    R["C15_RecipientState"] = np.where(proc["state_ok"].values, "PASS", "FAIL")
    R["Doc Type"] = proc["doc_type"].values
    R["Doc Issue"] = proc["doc_issue"].values
    R["Suggested SAP Action"] = proc["suggested_action"].values

    # --- C16 HSN (present in extract?) ---------------------------------
    hsn_col = next((c for c in df.columns if "hsn" in str(c).lower()), None)
    if hsn_col is not None:
        hsn = _txt(df, hsn_col)
        R["C16_HSN"] = np.where(hsn != "", "PASS", "FAIL")
    else:
        R["C16_HSN"] = "NA"

    # --- roll-up --------------------------------------------------------
    check_cols = list(CHECKS)
    def fails(row):
        return "; ".join(CHECKS[c] for c in check_cols
                         if str(row[c]).startswith("FAIL"))
    R["Failed Checks"] = R.apply(fails, axis=1)
    R["Fail Count"] = R[check_cols].apply(
        lambda r: sum(str(v).startswith("FAIL") for v in r), axis=1)
    R["Checklist Status"] = np.where(R["Fail Count"] == 0, "CLEAR",
                                     np.where(R["Fail Count"] >= 3, "HIGH",
                                              "REVIEW"))
    return R


def checklist_summary(R):
    """Per-check fail/pass/na counts for the dashboard."""
    rows = []
    for c, label in CHECKS.items():
        vals = R[c].astype(str)
        rows.append({
            "Check": c, "Description": label,
            "Pass": int(vals.str.startswith("PASS").sum()),
            "Fail": int(vals.str.startswith("FAIL").sum()),
            "NA": int(vals.str.startswith("NA").sum()),
            "Needs more data": CHECK_NOTES.get(c, ""),
        })
    return pd.DataFrame(rows)
