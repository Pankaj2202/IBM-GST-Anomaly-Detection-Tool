"""
GST Anomaly Detection Tool - Training Pipeline
JK Cement | Supply Chain Accounts Payable

Builds the labelled training set from ACL population + ARM anomaly flags,
trains the Random Forest, prints evaluation metrics, and saves the model +
metrics for the daily scoring run and the Excel report.
"""

import json
import joblib
import numpy as np
import pandas as pd

from gst_config import build_reference_pack
from gst_rules import standardize, run_rules
from gst_model import engineer_features, train_model, score

UP = "/mnt/user-data/uploads/"
ACL = UP + "1781600733092_ACL_Historical_Data.xlsx"
ARM = UP + "1781600733094_ARM_Invoices_Data_24-25.xlsm"
BP = UP + "1781600695484_Business_Place.xlsx"
TC = UP + "1781600721490_Tax_Code_List.xlsb"
VM = UP + "1781600713566_Unique_Vendor_List_with_GST_not_Code.xlsx"

ARM_TYPE = {
    "ARM-DollarValueDiff": "Amount mismatch (mean/median)",
    "ARM-FutureInvoice": "Future-dated invoice",
    "ARM-InvoicePatternMismatch": "Invoice-number pattern mismatch",
    "ARM-OldInvoice": "Stale/old invoice",
    "ARM-CurrMismatch": "Currency mismatch",
}


def build_labels():
    """Return {document_no -> anomaly_type} from the ARM workbook."""
    xl = pd.ExcelFile(ARM)
    doc_type = {}
    for sn in xl.sheet_names:
        d = xl.parse(sn)
        if "DocumentNo" not in d.columns or not len(d):
            continue
        docs = pd.to_numeric(d["DocumentNo"], errors="coerce").dropna().astype("int64")
        for dn in docs.unique():
            doc_type.setdefault(int(dn), ARM_TYPE.get(sn, sn))
    return doc_type


def main():
    print("Loading reference data ...")
    ref = build_reference_pack(BP, TC, VM)
    print(f"  vendors={len(ref['vendor_master'])}, "
          f"business_places={len(ref['valid_business_places'])}, "
          f"tax_codes={len(ref['valid_tax_codes'])}")

    print("Loading ACL population ...")
    acl = pd.read_excel(ACL, sheet_name="Consolidated_Data")
    acl.columns = ["Scripts", "CoCd", "Account", "VendorName", "Reference", "PK",
                   "DocDate", "DocAmt", "DocCurr", "PstngDate", "DocumentNo",
                   "Type", "LocAmt", "LocCurr", "ClrngDoc", "Clearing",
                   "DueDate", "Year", "Text", "Extra"]
    acl["DocumentNo"] = pd.to_numeric(acl["DocumentNo"], errors="coerce")
    acl = acl[acl["DocumentNo"].notna()].copy()
    print(f"  population rows: {len(acl)}")

    print("Building anomaly labels from ARM ...")
    labels = build_labels()
    acl["is_anomaly"] = acl["DocumentNo"].astype("int64").isin(labels).astype(int)
    acl["anomaly_type"] = acl["DocumentNo"].astype("int64").map(labels).fillna("None")
    print(f"  anomalies: {acl['is_anomaly'].sum()} "
          f"({acl['is_anomaly'].mean()*100:.1f}% of population)")

    print("Standardizing + running rule engine ...")
    std = standardize(acl)
    std["is_anomaly"] = acl["is_anomaly"].values
    std["anomaly_type"] = acl["anomaly_type"].values
    ruled = run_rules(std, ref)

    print("Engineering features ...")
    feat = engineer_features(ruled, ref)

    print("Training Random Forest ...")
    model, metrics = train_model(feat, label_col="is_anomaly")

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    print(f"Train/Test: {metrics['n_train']} / {metrics['n_test']}")
    print(f"Accuracy : {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall   : {metrics['recall']:.3f}")
    print(f"F1       : {metrics['f1']:.3f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.3f}")
    print(f"Confusion matrix [ [TN FP] [FN TP] ]: {metrics['confusion_matrix']}")
    print("\nTop features:")
    for f, imp in metrics["feature_importance"][:10]:
        print(f"  {f:<22} {imp:.3f}")

    print("\nScoring full population ...")
    scored = score(feat, model)

    joblib.dump({"model": model, "ref_meta": {
        "vendors": len(ref["vendor_master"]),
        "business_places": len(ref["valid_business_places"]),
        "tax_codes": len(ref["valid_tax_codes"]),
    }}, "gst_rf_model.joblib")

    metrics_save = {k: v for k, v in metrics.items() if k != "report"}
    metrics_save["feature_importance"] = [
        [f, float(i)] for f, i in metrics["feature_importance"]]
    with open("model_metrics.json", "w") as fh:
        json.dump(metrics_save, fh, indent=2)

    scored.to_pickle("scored_population.pkl")
    acl[["DocumentNo", "anomaly_type"]].to_pickle("anomaly_types.pkl")
    print("Saved: gst_rf_model.joblib, model_metrics.json, scored_population.pkl")
    return scored, metrics


if __name__ == "__main__":
    main()
