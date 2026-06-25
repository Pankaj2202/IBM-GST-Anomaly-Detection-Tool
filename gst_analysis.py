"""
GST Anomaly Detection Tool - Unified Analysis Engine
JK Cement | Supply Chain Accounts Payable

ONE entry point - run_full_analysis(df, ftype) - that runs every applicable
technique in a single pass and returns a structured result the Dashboard,
Batch-Scan and Analyst tabs all draw from. No tab re-computes anything.
"""

import numpy as np
import pandas as pd


def run_full_analysis(df, ftype, ref=None, score_frame=None):
    """ftype: 'recon' or 'posted_invoice'. Returns one results dict."""
    if ftype == "recon":
        return _analyse_recon(df)
    return _analyse_posted(df, score_frame)


def _analyse_recon(df):
    from gst_recon2b import reconcile, summarize
    from gst_checklist import run_checklist, checklist_summary
    from gst_assistant import build_findings, narrative_summary
    from gst_advanced import (benford_first_digit, vendor_network,
                              duplicate_invoices, round_threshold)

    rec = reconcile(df)
    by_type, by_band, tot = summarize(rec)
    chk = run_checklist(df)
    csum = checklist_summary(chk)
    findings = build_findings(df)
    fr = findings["fraud"]

    benford = benford_first_digit(df.get("InvoiceValue(PR)", pd.Series(dtype=float)))
    nflags, clusters = vendor_network(df, "SupplierGSTIN(PR)", "SupplierName(PR)")
    dup = duplicate_invoices(df, "SupplierGSTIN(PR)", "DocumentNumber(PR)",
                             "InvoiceValue(PR)", "DocumentDate(PR)")
    rt = round_threshold(pd.to_numeric(df.get("InvoiceValue(PR)"), errors="coerce"))

    # one combined per-invoice validation table
    combined = pd.DataFrame(index=df.index)
    combined["Supplier"] = fr["Supplier"]
    combined["Invoice No"] = fr["Invoice No"]
    combined["ITC at Risk"] = fr["ITC at Risk"]
    combined["Mismatch"] = rec["Mismatch Type"]
    combined["Checklist"] = chk["Checklist Status"]
    combined["Fails"] = chk["Fail Count"]
    combined["Fraud Score"] = fr["Fraud Risk Score"]
    combined["Risk"] = fr["Risk Level"]
    combined["Net.Risk"] = nflags["network_risk"].values
    combined["Duplicate"] = np.where(dup["exact_duplicate"], "Exact",
                                     np.where(dup["split_invoice"], "Split", ""))
    combined["Doc Type"] = chk.get("Doc Type", "")
    combined["SAP Action"] = chk.get("Suggested SAP Action", "")
    combined["Indicators"] = fr["Indicators"]
    combined = combined.sort_values(["Fraud Score", "ITC at Risk"], ascending=False)

    return {
        "kind": "recon", "rows": len(df),
        "total_itc_at_risk": tot["total_itc_at_risk"], "high_itc": tot["high_itc"],
        "by_type": by_type, "checklist_summary": csum,
        "fake_candidates": findings["fake_invoice_candidates"],
        "fake_itc": findings["fake_invoice_itc"],
        "top_vendors": findings["top_risk_vendors"],
        "narrative": narrative_summary(findings),
        "findings": findings,
        "benford": benford, "clusters": clusters,
        "n_exact_dup": int(dup["exact_duplicate"].sum()),
        "n_split": int(dup["split_invoice"].sum()),
        "n_round": int(rt["round_number"].sum()),
        "n_threshold": int(rt["just_under_threshold"].sum()),
        "high_network": int((nflags["network_risk"] >= 0.4).sum()),
        "combined": combined,
    }


def _analyse_posted(df, score_frame):
    from gst_advanced import (benford_first_digit, duplicate_invoices,
                              round_threshold, isolation_forest)
    from gst_model import FEATURES

    scored = score_frame(df)
    hi = int((scored["risk_band"] == "HIGH").sum())
    md = int((scored["risk_band"] == "MEDIUM").sum())

    iso = isolation_forest(scored, FEATURES)
    scored["iforest"] = iso.values
    benford = benford_first_digit(scored["doc_amount"])
    dup = duplicate_invoices(scored, "vendor_code", "invoice_no",
                             "doc_amount", "doc_date")
    rt = round_threshold(scored["doc_amount"])

    combined = pd.DataFrame(index=scored.index)
    combined["SAP Doc No"] = scored["document_no"]
    combined["Vendor"] = scored["vendor_name"]
    combined["Invoice No"] = scored["invoice_no"]
    combined["Amount"] = scored["doc_amount"]
    combined["Risk Score"] = scored["risk_score"].round(3)
    combined["Band"] = scored["risk_band"].astype(str)
    combined["Anomaly(iso)"] = scored["iforest"].round(3)
    combined["Duplicate"] = np.where(dup["exact_duplicate"], "Exact",
                                     np.where(dup["split_invoice"], "Split", ""))
    combined["Reasons"] = scored["rule_reasons"]
    combined = combined.sort_values("Risk Score", ascending=False)

    return {
        "kind": "posted_invoice", "rows": len(scored),
        "high": hi, "medium": md, "low": len(scored) - hi - md,
        "benford": benford,
        "n_exact_dup": int(dup["exact_duplicate"].sum()),
        "n_split": int(dup["split_invoice"].sum()),
        "n_round": int(rt["round_number"].sum()),
        "n_threshold": int(rt["just_under_threshold"].sum()),
        "scored": scored, "combined": combined,
    }
