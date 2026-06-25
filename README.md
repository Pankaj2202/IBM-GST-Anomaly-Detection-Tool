# GST Anomaly Detection Tool
**JK Cement | Supply Chain Accounts Payable**

A hybrid rule-engine + machine-learning tool that flags GST invoices with a high
risk of error or fraudulent Input Tax Credit (ITC) and **ranks them for human
review**. It replaces slow, error-prone manual checking with a prioritized daily
queue — it does **not** auto-reject invoices or label any vendor as fraudulent.

---

## 1. What it does

For every invoice it produces:
- A set of **deterministic rule flags** (auditable, each maps to a business rule).
- A single **Risk Score (0–1)** from a Random Forest model.
- A **Risk Band** — HIGH / MEDIUM / LOW — so reviewers work top-down by priority.

## 2. Architecture

```
Daily posted invoices ─┐
                       ├─► standardize ─► RULE ENGINE ─► feature engineering ─► RANDOM FOREST ─► Risk Score + Band ─► Review Queue
Reference masters ─────┘   (column map)   (5 validations)   (16 signals)         (300 trees)
   • Vendor Master
   • Business Place
   • Tax Code List
```

| File | Role |
|---|---|
| `gst_config.py` | Loads the 3 reference masters; central `THRESHOLDS`. |
| `gst_rules.py` | Column standardization + the deterministic validation rules. |
| `gst_model.py` | Feature engineering + Random Forest train/score. |
| `train.py` | Builds labels from ACL+ARM, trains, evaluates, saves the model. |
| `score_daily.py` | Scores a **new** daily file → prioritized review queue. |
| `build_report.py` | Builds the Excel showcase report. |

## 3. Data sources

- **ACL Historical Data** (≈101,880 invoices) — the population.
- **ARM Invoices Data** — historical anomaly flags across 4 live types
  (DollarValueDiff, FutureInvoice, InvoicePatternMismatch, OldInvoice).
  The **training label is 1** if an invoice's SAP document number appears in
  any ARM anomaly sheet, else 0.
- **Vendor Master / Business Place / Tax Code** — reference masters the rule
  engine validates against.

## 4. Validation algorithms (rule engine)

| # | Rule | Logic |
|---|---|---|
| R1 | Invoice No. | Present; vendor exists in active vendor master (by name or GSTIN). |
| R2 | Invoice Date | Not a future date vs posting date / today. |
| R3 | Invoice Amount | Deviation from the vendor's own historical median (mean/median %). |
| R4 | Tax Code | Valid code; consistent with Inter (IGST) / Intra (CGST+SGST). |
| R5 | Business Place | Valid; maps to a known JK Cement recipient GSTIN. |
| R6 | Filing Delay | Stale / old invoices beyond the ageing threshold (default 180d). |
| R7 | Round Amount | Large exact-round amounts (soft signal). |

All thresholds live in `THRESHOLDS` in `gst_config.py` — tune in one place.

## 5. The model

- **Random Forest Classifier** — 300 trees, `max_depth=18`, `class_weight="balanced"`
  to handle the ~17% anomaly rate.
- **16 engineered features** including filing delay, amount-vs-vendor-median %,
  vendor turnover & transaction count (turnover proxy), active-master membership,
  and invoice-number pattern features.
- Chosen for: handling non-linear feature interactions, robustness to class
  imbalance, and **feature-importance interpretability**.

### Performance (25% hold-out test)

| Metric | Value |
|---|---|
| Accuracy | 90.2% |
| Precision | 67.8% |
| **Recall (anomaly catch-rate)** | **81.5%** |
| F1 | 0.74 |
| ROC-AUC | 0.951 |

Recall is prioritized: in fraud screening, missing a real anomaly (false
negative) costs more than a false alarm a reviewer can dismiss.

## 6. How to run

```bash
pip install -r requirements.txt

# (Re)train from history — produces gst_rf_model.joblib + metrics
python train.py

# Score a new daily posted-invoice file
python score_daily.py  posted_invoices_DDMMYY.xlsx   ->  daily_review_queue.xlsx

# Rebuild the Excel showcase report
python build_report.py
```

The daily file only needs the usual SAP/ACL columns (Reference, DocDate, DocAmt,
PstngDate, DocumentNo, Account, VendorName, Type …). Column names are
auto-mapped; missing optional fields (tax code, business place, GSTIN) simply
skip those specific checks.

## 7. Important limitations — please read

1. **Risk, not verdict.** The labels are *anomalies / exceptions*, not
   court-proven fraud. A HIGH score means "review this first," not "this is
   fraud." Every flagged item needs human confirmation before any vendor action.
2. **Some features align with the rule definitions** (e.g. amount deviation, date
   logic). The model's real added value is combining weak signals into one
   calibrated score and surfacing items the fixed thresholds miss — not inventing
   a brand-new fraud signal.
3. **GSTR-2B vs PR reconciliation** (the `GST_Mismatch_Consolidated` file) is a
   richer fraud signal (claimed ITC absent from supplier filings). It is **not**
   yet wired into the model because its grain differs from the invoice
   population — see next steps.

## 8. GSTIN validation & the 12-point checklist (in the 2B vs PR Recon tab)

**GSTIN validation** (`gstin_validator.py`) runs in two layers:
- *Offline* — structural + mod-36 check-digit validation. Catches fabricated or
  mistyped GSTINs instantly, no internet, no cost. (Validated against JK's own
  31 GSTINs: 100% pass.)
- *Live (optional)* — `verify_live()` is a ready adapter for a GST Suvidha
  Provider / third-party API to confirm a GSTIN is registered & ACTIVE and read
  the legal name. Supply your endpoint + key; there is no free government API.

**12-point process checklist** (`gst_checklist.py`) runs on the 2B-vs-PR extract
and marks each invoice PASS / FAIL / NA per check:

| # | Check | Auto? |
|---|---|---|
| 1 | Single SAP doc per invoice (FLYASH exempt) | yes |
| 2 | Invoice complete (no mandatory field blank) | yes |
| 3 | Titled TAX INVOICE/CREDIT NOTE + JK Cement name | name only (title needs OCR) |
| 4 | Correct invoice date (not future) | yes |
| 5 | Correct invoice number (present, 2B=PR) | yes |
| 6 | Correct invoice amount (PR value = 2B value) | yes |
| 7 | Correct business place / recipient GSTIN (JK) | yes |
| 8 | Correct supplier GSTIN (valid, 2B=PR) | yes |
| 9 | Correct Inter/Intra tax vs supplier-recipient state | yes |
| 10 | Correct eligible/ineligible ITC treatment | yes (PO tax code needs PO data) |
| 11 | Correct reverse/forward charge treatment | yes |
| 12 | Correct invoice description (PO / invoice copy) | needs PO master or OCR |

To fully enable checks 3 and 12, provide the **PO master** (PO line items) and/or
run **OCR** on the invoice copies.

## 9. Advanced forensic analytics (📊 Advanced Analytics tab)

Market-grade techniques used by tax authorities (DGARM/DGGI) and forensic
auditors, in `gst_advanced.py`:

| Technique | Catches |
|---|---|
| **Benford's Law** (first-digit + Nigrini MAD) | Fabricated / engineered amounts |
| **Isolation Forest** (unsupervised) | Novel outliers the supervised model never saw |
| **Vendor-network signals** | Shell-company / circular-trading footprint — one PAN → many GSTINs, one GSTIN → many names, structurally invalid GSTINs |
| **Duplicate & split-invoice** | Double-posting and threshold-evading split invoicing |
| **Round-number / threshold-gaming** | Amounts engineered to round figures or just under approval limits |

The **AI Analyst** also fuses these into a single composite *fake-invoice /
tax-evasion* score per invoice.

## 10. Suggested next steps

- Integrate **GSTR-2B vs PR mismatch** (real fake-ITC signal) as a second model
  using the human-validated `Error Category` labels already in that file.
- Add **tax-amount recomputation** (R4 deep check): recompute expected GST from
  the tax-code rate and taxable value, flag mismatches.
- Add **GSTIN checksum + state-vs-business-place** structural validation.
- Schedule `score_daily.py` against the daily SAP extract and route the HIGH
  queue to the Hold Team mailbox.
