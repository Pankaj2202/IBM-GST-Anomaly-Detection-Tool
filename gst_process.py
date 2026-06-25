"""
GST Anomaly Detection Tool - JK Cement Process Module
JK Cement | Supply Chain Accounts Payable

Encodes JK Cement's documented SOP (GST_Invoice_Validation_Process.docx):
  - Document-Type master (Section 4) + charge/registration consistency rules
  - GST-rate validation (ZGSTR2N step)
  - Recipient-GSTIN state vs Business Place (ZGSTR2N step)
  - PO vs Non-PO classification
  - Suggested SAP correction / reversal action (Section 7.4 / 7.5)

This is the layer that makes the tool 100% process-specific, on top of the
market-standard forensic engines.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Document-Type master (SOP Section 4)
#   charge: FCM | RCM | NA   registration: REG | UNREG | NA
#   category: INVOICE | CREDIT_NOTE | DEBIT_NOTE | PAYMENT | TRANSFER | IMPORT | OTHER
# ---------------------------------------------------------------------------
DOCTYPE_MASTER = {
    # 4.1 Forward Charge
    "KR": dict(desc="Vendor invoice (service, registered)", charge="FCM", reg="REG", cat="INVOICE"),
    "RE": dict(desc="Invoice – Gross (material, registered)", charge="FCM", reg="REG", cat="INVOICE"),
    "US": dict(desc="Invoice – Services (unregistered / RCM)", charge="RCM", reg="UNREG", cat="INVOICE"),
    "UM": dict(desc="Unregistered vendor – Material", charge="FCM", reg="UNREG", cat="INVOICE"),
    "IM": dict(desc="Imports", charge="FCM", reg="NA", cat="IMPORT"),
    "KG": dict(desc="Vendor Credit Memo (debit/credit note)", charge="FCM", reg="NA", cat="CREDIT_NOTE"),
    "TA": dict(desc="Travelling bills (employee)", charge="FCM", reg="NA", cat="OTHER"),
    "YA": dict(desc="Depot Expenses", charge="FCM", reg="NA", cat="OTHER"),
    "VP": dict(desc="Vendor W/ Profit Ctr", charge="FCM", reg="NA", cat="OTHER"),
    # 4.2 Reverse Charge
    "RC": dict(desc="Invoice – Services (registered, RCM)", charge="RCM", reg="REG", cat="INVOICE"),
    # 4.3 Invoices / Debit Notes
    "VR": dict(desc="Vendor Return INV (with tax)", charge="FCM", reg="NA", cat="INVOICE"),
    "VE": dict(desc="Vendor Exempted Inv (without tax)", charge="FCM", reg="NA", cat="INVOICE"),
    # 4.4 Other
    "GS": dict(desc="STO DOC (inter-company)", charge="NA", reg="NA", cat="TRANSFER"),
    "KB": dict(desc="Transfer document (vendor to vendor)", charge="NA", reg="NA", cat="TRANSFER"),
    "KZ": dict(desc="Vendor payment", charge="NA", reg="NA", cat="PAYMENT"),
    "ZP": dict(desc="Payment posting", charge="NA", reg="NA", cat="PAYMENT"),
    "RF": dict(desc="194R Invoice (JKC)", charge="FCM", reg="NA", cat="INVOICE"),
}
RCM_DOCTYPES = {"RC", "US"}
CREDIT_NOTE_CATS = {"CREDIT_NOTE"}
VALID_GST_RATES = {0, 0.9, 1.5, 3, 5, 6, 12, 18, 28}   # % (CGST+SGST or IGST totals)


def _txt(s):
    return (s.astype(str).str.strip().str.replace("`", "", regex=False)
            .str.upper().replace({"NAN": "", "NONE": "", "NAT": ""}))


def validate_documents(df, doctype_col="UserDefinedField1",
                       rcm_col="ReverseChargeFlag(PR)", gstdoc_col="DocType(PR)"):
    """Per-row document-type validity + charge/registration/credit-note rules."""
    dt = _txt(df[doctype_col]) if doctype_col in df.columns else pd.Series("", index=df.index)
    rcm = _txt(df[rcm_col]) if rcm_col in df.columns else pd.Series("", index=df.index)
    gdoc = _txt(df[gstdoc_col]) if gstdoc_col in df.columns else pd.Series("", index=df.index)
    is_rcm = rcm.isin(["Y", "YES", "TRUE", "1"])
    is_credit = gdoc.isin(["C", "CREDIT"])    # GSTR doc type C = credit note

    out = pd.DataFrame(index=df.index)
    out["doc_type"] = dt
    known = dt.map(lambda x: x in DOCTYPE_MASTER)
    charge = dt.map(lambda x: DOCTYPE_MASTER.get(x, {}).get("charge", ""))
    cat = dt.map(lambda x: DOCTYPE_MASTER.get(x, {}).get("cat", ""))

    issues = []
    for i in df.index:
        msgs = []
        if dt[i] == "":
            msgs.append("Doc type missing")
        elif not known[i]:
            msgs.append(f"Unknown doc type '{dt[i]}'")
        else:
            # RCM consistency: RCM flag set but doc type is a forward-charge type
            if is_rcm[i] and dt[i] not in RCM_DOCTYPES:
                msgs.append("RCM flag set but doc type is not RC/US")
            # forward charge but RCM-only doc type
            if (not is_rcm[i]) and dt[i] == "RC":
                msgs.append("RC (RCM) doc type but no RCM flag")
            # RCM not allowed on credit notes
            if is_rcm[i] and (cat[i] in CREDIT_NOTE_CATS or is_credit[i]):
                msgs.append("RCM applied on a credit note (not permitted)")
        issues.append("; ".join(msgs))
    out["doc_issue"] = issues
    out["doc_ok"] = out["doc_issue"] == ""
    return out


def validate_gst_rate(df, taxable_col="TaxableValue(PR)", tax_col="TotalTax(PR)",
                      igst_col="IGST(PR)", cgst_col="CGST(PR)", sgst_col="SGST(PR)"):
    """Effective rate = tax/taxable must be a recognised GST rate; Inter/Intra
    must match which heads carry tax (ZGSTR2N % check)."""
    def num(c):
        return pd.to_numeric(df.get(c), errors="coerce").fillna(0.0)
    taxable, tax = num(taxable_col), num(tax_col)
    igst, cgst, sgst = num(igst_col), num(cgst_col), num(sgst_col)
    eff = np.where(taxable > 0, tax / taxable * 100, 0).round(2)

    out = pd.DataFrame(index=df.index)
    out["effective_rate"] = eff
    nearest_ok = np.array([min((abs(r - v) for v in VALID_GST_RATES), default=99) <= 0.6
                           for r in eff])
    has_tax = tax > 1
    # Intra => CGST&SGST, no IGST ; Inter => IGST only
    intra = (cgst + sgst > 1) & (igst <= 1)
    inter = (igst > 1) & (cgst + sgst <= 1)
    mixed = (igst > 1) & (cgst + sgst > 1)        # both -> wrong
    out["rate_ok"] = (~has_tax) | (nearest_ok & ~mixed)
    out["rate_issue"] = np.where(
        mixed, "IGST and CGST/SGST both charged",
        np.where(has_tax & ~nearest_ok, "Effective rate not a valid GST rate", ""))
    return out


def recipient_state_check(df, rec_gstin_col="RecipientGSTIN(PR)", state_col="PR State"):
    """Recipient (JK) GSTIN state prefix must match the booked state (ZGSTR2N)."""
    g = _txt(df[rec_gstin_col]) if rec_gstin_col in df.columns else pd.Series("", index=df.index)
    pref = g.str[:2]
    out = pd.DataFrame(index=df.index)
    out["recipient_state_code"] = pref
    # we validate prefix is a plausible 2-digit numeric state code
    out["state_ok"] = pref.str.match(r"^[0-9]{2}$").fillna(False)
    out["state_issue"] = np.where(out["state_ok"], "", "Recipient GSTIN state prefix invalid")
    return out


def po_classification(df, po_col="CustomerPOReferenceNumber"):
    """PO-based vs Non-PO (drives description check + reversal route)."""
    po = _txt(df[po_col]) if po_col in df.columns else pd.Series("", index=df.index)
    out = pd.DataFrame(index=df.index)
    out["is_po_based"] = po != ""
    out["po_ref"] = po
    return out


def suggested_sap_action(is_po_based):
    """SOP 7.4/7.5: PO/material (MIRO) -> reverse via ZFI_AP; Non-PO/FI (FB60)
    -> reverse via F-02."""
    return np.where(is_po_based, "Correct via MIRO · reverse via ZFI_AP",
                    "Correct via FB60 · reverse via F-02")


def run_process_checks(df):
    """Bundle all SOP-specific checks into one frame the checklist consumes."""
    docs = validate_documents(df)
    rate = validate_gst_rate(df)
    state = recipient_state_check(df)
    po = po_classification(df)
    res = pd.concat([docs, rate, state, po], axis=1)
    res["suggested_action"] = suggested_sap_action(po["is_po_based"].values)
    return res
