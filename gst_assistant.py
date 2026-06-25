"""
GST Anomaly Detection Tool - AI Analyst Engine
JK Cement | Supply Chain Accounts Payable

Three capabilities that power the conversational "Ask the GST Analyst" tab:

  1. fraud_indicators()  - a composite FAKE-INVOICE / TAX-EVASION risk score per
     invoice, fusing the deterministic signals (2B-missing ITC, invalid GSTIN,
     duplicate posting, ineligible ITC, tax-type mismatch). Auditable.
  2. build_findings()    - rolls everything into a structured findings dict.
  3. answer()            - natural-language Q&A. Works OFFLINE (intent-based,
     grounded in the computed numbers); if an LLM key is configured it routes
     to the LLM for free-form conversation, still grounded in the findings so
     it cannot invent figures.

Design rule: the LLM never computes tax/fraud itself - it only explains numbers
this engine already produced. That keeps every answer traceable.
"""

import os
import numpy as np
import pandas as pd

from gst_recon2b import reconcile, summarize
from gst_checklist import run_checklist, checklist_summary

# weights for the composite fake-invoice / evasion score
WEIGHTS = {
    "itc_2b_missing": 0.40,    # ITC claimed, supplier never filed in 2B
    "invalid_gstin": 0.25,     # supplier GSTIN fails checksum (fabricated/typo)
    "duplicate_posting": 0.15, # same invoice booked as multiple SAP docs
    "ineligible_itc": 0.15,    # ITC availed on an ineligible head
    "tax_type_mismatch": 0.10, # Inter/Intra applied wrongly
    "rcm_mismatch": 0.10,      # reverse-charge treatment differs 2B vs PR
}


def fraud_indicators(df):
    """Composite fake-invoice / tax-evasion risk per invoice (0-1) + reasons."""
    rec = reconcile(df)
    chk = run_checklist(df)
    n = len(df)

    sig = pd.DataFrame(index=df.index)
    sig["itc_2b_missing"] = (rec["Mismatch Type"]
                             .str.startswith("2B-Missing")).astype(int)
    sig["invalid_gstin"] = (chk["C8_SupplierGSTIN"] == "FAIL").astype(int)
    sig["duplicate_posting"] = (chk["C1_SingleDocPerInvoice"] == "FAIL").astype(int)
    sig["ineligible_itc"] = (chk["C10_Eligibility"] == "FAIL").astype(int)
    sig["tax_type_mismatch"] = (chk["C9_InterIntraTax"] == "FAIL").astype(int)
    sig["rcm_mismatch"] = (chk["C11_RCM_FCM"] == "FAIL").astype(int)

    score = sum(sig[k] * w for k, w in WEIGHTS.items())
    score = score.clip(upper=1.0)

    label = {
        "itc_2b_missing": "ITC not in supplier 2B",
        "invalid_gstin": "Invalid supplier GSTIN",
        "duplicate_posting": "Duplicate posting",
        "ineligible_itc": "Ineligible ITC availed",
        "tax_type_mismatch": "Wrong Inter/Intra tax",
        "rcm_mismatch": "RCM/FCM mismatch",
    }
    reasons = sig.apply(lambda r: "; ".join(label[k] for k in WEIGHTS if r[k]),
                        axis=1)

    out = pd.DataFrame(index=df.index)
    nm = rec["Supplier Name"].fillna("").astype(str).str.strip()
    gst = rec["Supplier GSTIN (PR)"].fillna("").astype(str)
    out["Supplier"] = nm.where(nm != "", gst).replace("", "(unnamed)")
    out["Supplier GSTIN"] = gst
    out["Invoice No"] = rec["Invoice No (PR)"]
    out["ITC at Risk"] = rec["ITC at Risk"]
    out["Fraud Risk Score"] = score.round(3)
    out["Risk Level"] = np.where(score >= 0.5, "HIGH",
                                 np.where(score >= 0.25, "MEDIUM", "LOW"))
    out["Indicators"] = reasons
    return out


def build_findings(df):
    """Run every engine and return a structured findings dict for chat/summary."""
    rec = reconcile(df)
    by_type, by_band, tot = summarize(rec)
    chk = run_checklist(df)
    csum = checklist_summary(chk)
    fr = fraud_indicators(df)

    vendor_risk = (fr.groupby("Supplier")
                   .agg(ITC_at_Risk=("ITC at Risk", "sum"),
                        Invoices=("ITC at Risk", "size"),
                        Avg_Fraud_Score=("Fraud Risk Score", "mean"))
                   .sort_values("ITC_at_Risk", ascending=False).head(15)
                   .reset_index())

    return {
        "rows": len(df),
        "total_itc_at_risk": float(tot["total_itc_at_risk"]),
        "high_itc": float(tot["high_itc"]),
        "mismatch_by_type": by_type,
        "checklist_summary": csum,
        "fraud": fr,
        "fake_invoice_candidates": int((fr["Risk Level"] == "HIGH").sum()),
        "fake_invoice_itc": float(fr.loc[fr["Risk Level"] == "HIGH", "ITC at Risk"].sum()),
        "top_risk_vendors": vendor_risk,
    }


def narrative_summary(f):
    """Plain-language executive summary built from the findings (no LLM)."""
    cr = lambda x: f"Rs {x/1e7:,.2f} Cr" if x >= 1e7 else f"Rs {x:,.0f}"
    top_type = f["mismatch_by_type"].iloc[0] if len(f["mismatch_by_type"]) else None
    fails = f["checklist_summary"].sort_values("Fail", ascending=False)
    top_fail = fails.iloc[0] if len(fails) else None
    lines = [
        f"Analysed {f['rows']:,} records.",
        f"Total ITC at risk is {cr(f['total_itc_at_risk'])}, of which "
        f"{cr(f['high_itc'])} sits in HIGH-risk records.",
    ]
    if top_type is not None:
        lines.append(f"The largest exposure is '{top_type['Mismatch Type']}' "
                     f"({int(top_type['Invoices']):,} invoices, "
                     f"{cr(top_type['ITC_at_Risk'])}).")
    lines.append(f"{f['fake_invoice_candidates']:,} invoices are flagged as "
                 f"fake-invoice / evasion candidates ({cr(f['fake_invoice_itc'])} "
                 f"ITC), combining unsupported ITC, invalid GSTINs, duplicates "
                 f"and ineligible claims.")
    if top_fail is not None and top_fail["Fail"] > 0:
        lines.append(f"The most-failed process check is '{top_fail['Description']}' "
                     f"({int(top_fail['Fail']):,} invoices).")
    if len(f["top_risk_vendors"]):
        v = f["top_risk_vendors"].iloc[0]
        lines.append(f"Highest-exposure supplier: {v['Supplier']} "
                     f"({cr(v['ITC_at_Risk'])} across {int(v['Invoices'])} invoices).")
    lines.append("These are risk indicators for review, not confirmed fraud - "
                 "each needs human confirmation before any vendor action.")
    return " ".join(lines)


# ---------------------------------------------------------------------------
# Offline intent-based Q&A (works with no API key)
# ---------------------------------------------------------------------------
def answer_offline(q, f):
    ql = q.lower()
    cr = lambda x: f"Rs {x/1e7:,.2f} Cr" if x >= 1e7 else f"Rs {x:,.0f}"

    if any(w in ql for w in ["summary", "overview", "summarise", "summarize", "brief"]):
        return narrative_summary(f)
    if any(w in ql for w in ["top vendor", "which vendor", "worst vendor",
                             "highest vendor", "supplier", "by vendor"]):
        v = f["top_risk_vendors"].head(8)
        rows = "\n".join(f"  {i+1}. {r['Supplier']} — {cr(r['ITC_at_Risk'])} "
                         f"({int(r['Invoices'])} inv)" for i, r in v.iterrows())
        return "Top suppliers by ITC at risk:\n" + rows
    if any(w in ql for w in ["fake", "evasion", "fraud", "suspicious"]):
        return (f"{f['fake_invoice_candidates']:,} invoices are HIGH-risk "
                f"fake-invoice / evasion candidates, holding {cr(f['fake_invoice_itc'])} "
                f"of ITC. Drivers: unsupported ITC (not in supplier 2B), invalid "
                f"GSTINs, duplicate postings and ineligible claims. Review the "
                f"'Fake-invoice candidates' table for the ranked list.")
    if any(w in ql for w in ["2b", "not filed", "missing", "unsupported"]):
        mt = f["mismatch_by_type"]
        row = mt[mt["Mismatch Type"].str.startswith("2B-Missing")]
        if len(row):
            r = row.iloc[0]
            return (f"{int(r['Invoices']):,} invoices claim ITC that is absent "
                    f"from the supplier's GSTR-2B — {cr(r['ITC_at_Risk'])} at risk. "
                    f"This is the classic unsupported/fake-ITC exposure.")
    if "ineligible" in ql:
        cs = f["checklist_summary"]
        r = cs[cs["Check"] == "C10_Eligibility"].iloc[0]
        return f"{int(r['Fail']):,} invoices availed ITC on an ineligible head."
    if any(w in ql for w in ["invalid gstin", "fake gstin", "gstin"]):
        cs = f["checklist_summary"]
        r = cs[cs["Check"] == "C8_SupplierGSTIN"].iloc[0]
        return (f"{int(r['Fail']):,} invoices have a supplier GSTIN that fails "
                f"structural/check-digit validation (fabricated or mistyped).")
    if any(w in ql for w in ["duplicate", "double", "twice"]):
        cs = f["checklist_summary"]
        r = cs[cs["Check"] == "C1_SingleDocPerInvoice"].iloc[0]
        return (f"{int(r['Fail']):,} invoices appear posted as more than one SAP "
                f"document (FLYASH vendors are exempt).")
    if any(w in ql for w in ["checklist", "process", "checks", "compliance"]):
        cs = f["checklist_summary"].sort_values("Fail", ascending=False).head(5)
        rows = "\n".join(f"  • {r['Description']}: {int(r['Fail']):,} fail"
                         for _, r in cs.iterrows())
        return "Top process-checklist failures:\n" + rows
    if any(w in ql for w in ["total", "how much", "at risk", "exposure"]):
        return (f"Total ITC at risk: {cr(f['total_itc_at_risk'])} across "
                f"{f['rows']:,} records ({cr(f['high_itc'])} HIGH-risk).")
    return ("I can answer questions about: overall summary, total ITC at risk, "
            "top risk vendors, fake-invoice/evasion candidates, 2B-missing ITC, "
            "ineligible ITC, invalid GSTINs, duplicate postings, and the process "
            "checklist. Try e.g. 'top vendors by ITC at risk' or 'summarise the risks'.")


# ---------------------------------------------------------------------------
# Optional LLM backend (Anthropic / OpenAI). Used only if a key is configured.
# ---------------------------------------------------------------------------
def llm_configured(secrets=None):
    s = secrets or {}
    return bool(s.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY")
                or s.get("openai_api_key") or os.environ.get("OPENAI_API_KEY"))


def _context_block(f):
    cr = lambda x: f"Rs {x:,.0f}"
    mt = "\n".join(f"- {r['Mismatch Type']}: {int(r['Invoices'])} inv, "
                   f"{cr(r['ITC_at_Risk'])}" for _, r in f["mismatch_by_type"].iterrows())
    cs = "\n".join(f"- {r['Description']}: {int(r['Fail'])} fail / {int(r['NA'])} NA"
                   for _, r in f["checklist_summary"].iterrows())
    tv = "\n".join(f"- {r['Supplier']}: {cr(r['ITC_at_Risk'])} ({int(r['Invoices'])} inv)"
                   for _, r in f["top_risk_vendors"].head(10).iterrows())
    return (f"FINDINGS (already computed; use ONLY these numbers):\n"
            f"Records: {f['rows']}\nTotal ITC at risk: {cr(f['total_itc_at_risk'])}\n"
            f"HIGH-risk ITC: {cr(f['high_itc'])}\n"
            f"Fake-invoice/evasion candidates: {f['fake_invoice_candidates']} "
            f"({cr(f['fake_invoice_itc'])})\n\nMismatch by type:\n{mt}\n\n"
            f"Checklist:\n{cs}\n\nTop risk vendors:\n{tv}")


def answer_llm(q, f, secrets=None):
    """Free-form answer via LLM, grounded in the findings. Returns (text, ok)."""
    s = secrets or {}
    sys = ("You are a GST compliance analyst for JK Cement Accounts Payable. "
           "Answer ONLY from the FINDINGS provided. Never invent numbers. If the "
           "answer isn't in the findings, say so. Be concise and practical. Always "
           "treat flags as risk indicators for review, not confirmed fraud.")
    ctx = _context_block(f)
    prompt = f"{ctx}\n\nQUESTION: {q}"
    try:
        akey = s.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY")
        if akey:
            import anthropic
            cl = anthropic.Anthropic(api_key=akey)
            r = cl.messages.create(
                model=s.get("model", "claude-sonnet-4-6"),
                max_tokens=700, system=sys,
                messages=[{"role": "user", "content": prompt}])
            return "".join(b.text for b in r.content if b.type == "text"), True
        okey = s.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
        if okey:
            from openai import OpenAI
            cl = OpenAI(api_key=okey)
            r = cl.chat.completions.create(
                model=s.get("model", "gpt-4o-mini"),
                messages=[{"role": "system", "content": sys},
                          {"role": "user", "content": prompt}], max_tokens=700)
            return r.choices[0].message.content, True
    except Exception as e:
        return f"(LLM error, showing offline answer instead: {e})", False
    return "", False


def answer(q, f, secrets=None):
    """Main entry: LLM if configured (fallback to offline), else offline."""
    if llm_configured(secrets):
        text, ok = answer_llm(q, f, secrets)
        if ok:
            return text
        return (text + "\n\n" + answer_offline(q, f)) if text else answer_offline(q, f)
    return answer_offline(q, f)
