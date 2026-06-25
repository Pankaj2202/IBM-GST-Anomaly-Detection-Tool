"""
GST Anomaly Detection Tool - Daily Scoring CLI
JK Cement | Supply Chain Accounts Payable

Apply the trained model + rule engine to a NEW daily posted-invoice file and
write a prioritized review queue.

Usage:
    python score_daily.py  <posted_invoices.xlsx>  [sheet_name]  [out.xlsx]

The input file just needs the usual SAP/ACL columns (Reference, DocDate,
DocAmt, PstngDate, DocumentNo, Account, VendorName, Type ...). Column names
are auto-mapped; missing optional fields (tax_code, business_place,
vendor_gstin) simply skip those specific rule checks.
"""

import sys
import joblib
import pandas as pd

from gst_config import build_reference_pack, THRESHOLDS
from gst_rules import standardize, run_rules
from gst_model import engineer_features, score

UP = "/mnt/user-data/uploads/"
BP = UP + "1781600695484_Business_Place.xlsx"
TC = UP + "1781600721490_Tax_Code_List.xlsb"
VM = UP + "1781600713566_Unique_Vendor_List_with_GST_not_Code.xlsx"

OUT_COLS = {
    "document_no": "SAP Doc No", "vendor_code": "Vendor Code",
    "vendor_name": "Vendor Name", "invoice_no": "Invoice No",
    "doc_date": "Invoice Date", "pstng_date": "Posting Date",
    "doc_amount": "Amount", "filing_delay_days": "Filing Delay (d)",
    "amount_dev_pct": "Amt Dev %", "rule_flag_count": "Rule Flags",
    "rule_reasons": "Reasons", "risk_score": "Risk Score",
    "risk_band": "Risk Band",
}


def run(in_path, sheet=0, out_path="daily_review_queue.xlsx"):
    ref = build_reference_pack(BP, TC, VM)
    model = joblib.load("gst_rf_model.joblib")["model"]

    raw = pd.read_excel(in_path, sheet_name=sheet)
    std = standardize(raw)
    ruled = run_rules(std, ref)
    feat = engineer_features(ruled, ref)
    scored = score(feat, model)

    queue = (scored.sort_values("risk_score", ascending=False)
             [list(OUT_COLS)].rename(columns=OUT_COLS))
    queue.to_excel(out_path, index=False)

    n = len(scored)
    hi = (scored["risk_band"] == "HIGH").sum()
    md = (scored["risk_band"] == "MEDIUM").sum()
    print(f"Scored {n} invoices -> HIGH={hi}  MEDIUM={md}  LOW={n-hi-md}")
    print(f"Review queue written to {out_path}")
    return scored


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    args = sys.argv[1:]
    in_path = args[0]
    sheet = args[1] if len(args) > 1 else 0
    out_path = args[2] if len(args) > 2 else "daily_review_queue.xlsx"
    run(in_path, sheet, out_path)
