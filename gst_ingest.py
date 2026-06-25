"""
GST Anomaly Detection Tool - Smart Ingestion Layer
JK Cement | Supply Chain Accounts Payable

Makes the tool schema-agnostic: it reads any xlsx/xlsm/xlsb/csv, picks the
data sheet, maps columns by MEANING (synonyms + fuzzy match, not exact
headers), auto-detects whether the file is posted-invoice data or a 2B-vs-PR
reconciliation extract, and returns a transparent "data profile" so the user
sees exactly what was recognised. Nothing crashes on an unexpected layout - it
degrades and reports.
"""

import io
import difflib
import pandas as pd

# ---------------------------------------------------------------------------
# Canonical fields and their many possible header spellings (normalised).
# Add new synonyms here - detection improves everywhere automatically.
# ---------------------------------------------------------------------------
FIELD_SYNONYMS = {
    # posted-invoice fields
    "invoice_no": ["invoiceno", "invoicenumber", "invno", "reference", "billno",
                   "billnumber", "documentnumberpr", "documentnumber", "vendorinvoiceno",
                   "supplierinvoiceno", "invoicereference"],
    "doc_date": ["docdate", "documentdate", "invoicedate", "billdate",
                 "documentdatepr", "invdate"],
    "pstng_date": ["pstngdate", "postingdate", "glpostingdate", "postdate",
                   "accountingdate", "accountingvoucherdate"],
    "doc_amount": ["docamt", "documentamount", "invoiceamount", "amount",
                   "invoicevaluepr", "invoicevalue", "grossamount", "locamt", "value"],
    "document_no": ["documentno", "sapdocno", "accountingvouchernumber",
                    "vouchernumber", "fidocno", "sapdocumentno"],
    "vendor_name": ["vendorname", "suppliername", "suppliernamepr", "partyname",
                    "name", "vendor"],
    "vendor_code": ["account", "vendorcode", "suppliercode", "vendorno",
                    "vendoraccount", "lifnr"],
    "vendor_gstin": ["vendorgstin", "suppliergstin", "suppliergstinpr", "gstin",
                     "gstno", "gstinno", "suppliergstno"],
    "tax_code": ["taxcode", "mwskz", "taxcd"],
    "business_place": ["businessplace", "plantcode", "plant", "bupla", "branch"],
    "doc_type": ["doctype", "documenttype", "type", "blart", "doctypepr"],
    "co_code": ["cocd", "companycode", "company"],
    "posting_key": ["pk", "postingkey"],
    "text": ["text", "narration", "description", "remarks"],
}

# columns whose presence signals a GSTR-2B-vs-PR reconciliation extract
RECON_SIGNATURE = [
    "igst2b", "cgst2b", "sgst2b", "taxablevalue2b", "documentnumber2b",
    "documentnumberpr", "recipientgstin2b", "eligibilityindicator",
    "reversechargeflag2b", "totaltax2b",
]
# minimum canonical fields to be usable as a posted-invoice file
POSTED_MIN = ["invoice_no", "doc_amount"]


def _norm(s):
    return "".join(ch for ch in str(s).strip().lower() if ch.isalnum())


def read_any(file, filename=""):
    """Read xlsx/xlsm/xlsb/csv into a DataFrame, choosing the richest sheet."""
    name = (filename or getattr(file, "name", "") or "").lower()
    if name.endswith(".csv"):
        return pd.read_csv(file), "csv"
    engine = "pyxlsb" if name.endswith(".xlsb") else None
    xl = pd.ExcelFile(file, engine=engine)
    # pick the sheet with the most non-empty cells (skips empty/menu sheets)
    best, best_score, best_name = None, -1, None
    for sn in xl.sheet_names:
        d = xl.parse(sn)
        score = d.notna().sum().sum()
        if score > best_score:
            best, best_score, best_name = d, score, sn
    return best, best_name


def map_columns(df, fuzzy_cutoff=0.86):
    """
    Map df columns -> canonical fields. Two passes: exact-normalised synonym
    match, then fuzzy match for anything unresolved. Returns
    (mapping {df_col: canonical}, detail list).
    """
    norm_to_canon = {}
    for canon, syns in FIELD_SYNONYMS.items():
        for s in syns:
            norm_to_canon[_norm(s)] = canon
        norm_to_canon[_norm(canon)] = canon

    mapping, detail, used = {}, [], set()
    syn_keys = list(norm_to_canon)

    for col in df.columns:
        nc = _norm(col)
        canon, how, score = None, "", 0.0
        if nc in norm_to_canon:
            canon, how, score = norm_to_canon[nc], "exact", 1.0
        else:
            match = difflib.get_close_matches(nc, syn_keys, n=1, cutoff=fuzzy_cutoff)
            if match:
                canon, how, score = norm_to_canon[match[0]], "fuzzy", \
                    difflib.SequenceMatcher(None, nc, match[0]).ratio()
        # keep the first (best) column claimed for each canonical field
        if canon and canon not in used:
            mapping[col] = canon
            used.add(canon)
            detail.append({"source": col, "mapped_to": canon,
                           "match": how, "confidence": round(score, 2)})
    return mapping, detail


def detect_file_type(df):
    """Return 'recon' | 'posted_invoice' | 'unknown' from the columns present."""
    norms = {_norm(c) for c in df.columns}
    recon_hits = sum(1 for s in RECON_SIGNATURE if s in norms)
    if recon_hits >= 4:
        return "recon", recon_hits
    mapping, _ = map_columns(df)
    canon = set(mapping.values())
    if all(f in canon for f in POSTED_MIN) and (
            "pstng_date" in canon or "document_no" in canon or "vendor_name" in canon):
        return "posted_invoice", len(canon)
    return "unknown", len(canon)


def profile(file, filename=""):
    """
    One call the app uses: read, detect, map. Returns a dict describing the
    file so the UI can show the user exactly what was recognised and route.
    """
    df, sheet = read_any(file, filename)
    ftype, strength = detect_file_type(df)
    mapping, detail = map_columns(df)
    canon = set(mapping.values())
    missing_posted = [f for f in ["invoice_no", "doc_date", "doc_amount",
                                  "pstng_date", "document_no", "vendor_name"]
                      if f not in canon]
    return {
        "df": df,
        "sheet": sheet,
        "rows": len(df),
        "cols": len(df.columns),
        "file_type": ftype,            # recon | posted_invoice | unknown
        "type_strength": strength,
        "mapping": mapping,            # {source_col: canonical}
        "mapping_detail": detail,      # list of dicts for display
        "fields_found": sorted(canon),
        "missing_posted_fields": missing_posted,
    }
