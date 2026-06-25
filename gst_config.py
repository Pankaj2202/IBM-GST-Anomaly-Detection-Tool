"""
GST Anomaly Detection Tool - Configuration & Reference Data Loader
JK Cement | Supply Chain Accounts Payable

Loads the three master/reference datasets the rule engine validates against:
  1. Vendor Master  (active vendor name + GSTIN)
  2. Business Place  (JK recipient GSTIN <-> business place code <-> state)
  3. Tax Code List   (tax code -> eligibility / inter-intra / GST rate)

All thresholds used by the rule engine live in THRESHOLDS so they can be
tuned in one place without touching detection logic.
"""

import pandas as pd
from pyxlsb import open_workbook

# ---------------------------------------------------------------------------
# Tunable thresholds  (single source of truth for the rule engine)
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "old_invoice_days": 180,        # filing delay above this = "old invoice" flag
    "amount_dev_pct": 50,           # % deviation from vendor median amount = outlier
    "min_vendor_history": 5,        # need >= N invoices before amount-deviation is trusted
    "round_amount_modulo": 10000,   # exact multiples of this on large invoices = soft flag
    "high_risk_score": 0.70,        # model probability >= this -> HIGH risk band
    "med_risk_score": 0.40,         # model probability >= this -> MEDIUM risk band
}

# GSTIN structural constants
GSTIN_LEN = 15
GSTIN_PAN_SLICE = slice(2, 12)     # chars 3-12 = PAN
GSTIN_STATE_SLICE = slice(0, 2)    # chars 1-2 = state code


# ---------------------------------------------------------------------------
# Reference data loaders
# ---------------------------------------------------------------------------
def load_business_place(path):
    """Return df[business_place, jk_gstin, state, state_code]."""
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={
        df.columns[0]: "business_place",
        df.columns[1]: "jk_gstin",
        df.columns[2]: "state",
    })
    df["business_place"] = df["business_place"].astype(str).str.strip()
    df["jk_gstin"] = df["jk_gstin"].astype(str).str.strip().str.upper()
    df["state_code"] = df["jk_gstin"].str[GSTIN_STATE_SLICE]
    return df[["business_place", "jk_gstin", "state", "state_code"]]


def load_tax_code(path):
    """Return df[tax_code, head, eligibility, inter_intra, cgst, sgst, igst, total_rate]."""
    with open_workbook(path) as wb:
        sheet_name = wb.sheets[0]
        with wb.get_sheet(sheet_name) as sheet:
            rows = [[c.v for c in row] for row in sheet.rows()]
    # header is on the 2nd row (row index 1)
    header = [str(h).strip() if h is not None else "" for h in rows[1]]
    data = rows[2:]
    df = pd.DataFrame(data, columns=header)
    df = df.rename(columns={
        "Tax code": "tax_code",
        "Tax Description": "description",
        "Head": "head",
        "Eligible / Ineligible": "eligibility",
        "Inter/Intra": "inter_intra",
        "CGST": "cgst", "SGST": "sgst", "IGST": "igst",
    })
    df = df[df["tax_code"].notna()].copy()
    df["tax_code"] = df["tax_code"].astype(str).str.strip().str.upper()
    for c in ["cgst", "sgst", "igst"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["total_rate"] = (df["cgst"] + df["sgst"] + df["igst"]).round(4)
    df["inter_intra"] = df["inter_intra"].astype(str).str.strip().str.title()
    df["eligibility"] = df["eligibility"].astype(str).str.strip()
    return df[["tax_code", "description", "head", "eligibility",
               "inter_intra", "cgst", "sgst", "igst", "total_rate"]]


def load_vendor_master(path):
    """Return df[vendor_name, vendor_gstin, vendor_name_norm] of active vendors."""
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={
        df.columns[0]: "vendor_name",
        df.columns[1]: "vendor_gstin",
    })
    df["vendor_name"] = df["vendor_name"].astype(str).str.strip()
    df["vendor_gstin"] = df["vendor_gstin"].astype(str).str.strip().str.upper()
    df["vendor_name_norm"] = (df["vendor_name"].str.upper()
                              .str.replace(r"[^A-Z0-9]", "", regex=True))
    return df.drop_duplicates()


def build_reference_pack(business_place_path, tax_code_path, vendor_master_path):
    """Convenience: load all three into a single dict consumed by the rule engine."""
    bp = load_business_place(business_place_path)
    tc = load_tax_code(tax_code_path)
    vm = load_vendor_master(vendor_master_path)
    return {
        "business_place": bp,
        "tax_code": tc,
        "vendor_master": vm,
        "valid_business_places": set(bp["business_place"]),
        "valid_jk_gstins": set(bp["jk_gstin"]),
        "bp_state_lookup": dict(zip(bp["business_place"], bp["state_code"])),
        "valid_tax_codes": set(tc["tax_code"]),
        "tax_code_lookup": tc.set_index("tax_code").to_dict("index"),
        "active_vendor_gstins": set(vm["vendor_gstin"]),
        "active_vendor_names": set(vm["vendor_name_norm"]),
    }
