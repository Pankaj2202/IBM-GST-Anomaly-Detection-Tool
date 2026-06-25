# Web App — Setup & Use

A browser-based validation UI for the GST Anomaly Detection Tool. Same engine
(rule checks + Random Forest) as the command-line version, wrapped in a point-
and-click interface where AP users upload a file and validate invoices.

---

## What the user sees

**Tab 1 — Batch scan & validate**
Upload the daily posted-invoice file → instant scored queue ranked worst-first →
KPI tiles (HIGH / MEDIUM / LOW) → an editable table where the reviewer sets a
**Decision** (Validated / Hold / Query) and types **Remarks** on each row →
one-click **Export** of the reviewed file.

**Tab 2 — Single-invoice check ("engine" mode)**
Paste one invoice's details (vendor, GSTIN, dates, amount, business place, tax
code) → instant verdict. Here the **deterministic rule checks drive the band**,
because a lone invoice has no vendor history for the ML model to lean on.

**Tab 3 — How to use** — in-app quick reference.

---

## Developer setup (one time)

```
GST_Anomaly_Detection_Tool/
├── app.py                       <- the web app
├── gst_config.py  gst_rules.py  gst_model.py
├── gst_rf_model.joblib          <- trained model (ships in the zip)
├── requirements.txt
└── reference/                   <- put the 3 master files here
    ├── Business_Place.xlsx
    ├── Tax_Code_List.xlsb
    └── Unique_Vendor_List_with_GST_not_Code.xlsx
```

1. **Install**
   ```bash
   cd GST_Anomaly_Detection_Tool
   pip install -r requirements.txt
   ```
2. **Reference files** — they're already in `./reference`. If your filenames
   differ, edit the four path lines at the top of `app.py`.
3. **Launch**
   ```bash
   streamlit run app.py
   ```
   A browser tab opens at `http://localhost:8501`. That's it.

---

## Letting the team use it (not just localhost)

`streamlit run app.py` only serves *your* machine. To give the AP team a real
URL, host it once:

| Option | How | Notes |
|---|---|---|
| **Internal server / VM** | Run `streamlit run app.py --server.port 8501 --server.address 0.0.0.0` on a shared Windows/Linux box; share `http://<that-machine-ip>:8501` | Best for JK Cement — data stays inside the network. |
| **Docker** | Wrap the folder in a container, deploy to your internal container host | Cleanest for IT to manage. |
| **Streamlit Community Cloud** | Push to a private repo, deploy | ❌ Avoid — this is client financial data; do **not** put it on a public cloud. |

**Data-governance note:** invoices, vendor masters and GSTINs are sensitive JK
Cement / client data. Host internally, behind the corporate network and the
usual access controls. Don't deploy to any public endpoint.

---

## Retraining

The web app reads `gst_rf_model.joblib`. To refresh the model on newer history,
run the CLI `python train.py` (it rewrites that file), then restart the app.
