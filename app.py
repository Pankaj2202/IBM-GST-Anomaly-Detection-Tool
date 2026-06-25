"""
GST Anomaly & Fraud Analytics Platform  (Streamlit)
JK Cement (client)  ·  IBM (service provider)

Run:  streamlit run app.py

Tabs:
  Dashboard               - summary of the LAST file analysed
  Batch Scan & Validate   - one smart upload: auto-detects the file type and runs
                            EVERYTHING applicable (ML scoring OR 2B-PR recon +
                            12-point checklist + GSTIN validation + Benford +
                            vendor-network + duplicates + fraud scoring)
  Ask the GST Analyst     - conversational analysis of the last file
  Single-invoice check    - one invoice on demand
"""

import io
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

from gst_config import build_reference_pack
from gst_rules import standardize, run_rules
from gst_model import engineer_features, score
from gst_analysis import run_full_analysis

BASE = Path(__file__).parent
REF = BASE / "reference"
BP_FILE = REF / "Business_Place.xlsx"
TC_FILE = REF / "Tax_Code_List.xlsb"
VM_FILE = REF / "Unique_Vendor_List_with_GST_not_Code.xlsx"
MODEL_FILE = BASE / "gst_rf_model.joblib"

st.set_page_config(page_title="GST Fraud Analytics | JK Cement × IBM",
                   page_icon="🛡️", layout="wide")

JK_RED, IBM_BLUE, NAVY = "#D81F2A", "#0F62FE", "#1F3A5F"
BAND_COLOR = {"HIGH": "#C0392B", "MEDIUM": "#E67E22", "LOW": "#27AE60"}


def _auth_gate():
    try:
        pw = st.secrets.get("app_password", None)
    except Exception:
        pw = None
    if not pw or st.session_state.get("authed"):
        return
    st.markdown("### 🔒 GST Fraud Analytics Platform")
    st.caption("JK Cement · Supply Chain Accounts Payable")
    entered = st.text_input("Password", type="password")
    if st.button("Sign in"):
        if entered == pw:
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("Incorrect password")
    st.stop()


_auth_gate()


@st.cache_resource(show_spinner="Loading reference masters + model …")
def load_assets():
    ref = build_reference_pack(str(BP_FILE), str(TC_FILE), str(VM_FILE))
    model = joblib.load(MODEL_FILE)["model"]
    return ref, model


def score_frame(raw_df):
    return score(engineer_features(run_rules(standardize(raw_df), ref), ref), model)


def to_excel_bytes(df, sheet="Sheet1"):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        df.to_excel(xl, index=False, sheet_name=sheet)
    return buf.getvalue()


def inr(x):
    try:
        return f"₹{x/1e7:,.2f} Cr" if x >= 1e7 else f"₹{x:,.0f}"
    except Exception:
        return "—"


def _logo(path, fallback, color, role):
    p = BASE / "assets" / path
    if p.exists():
        import base64
        b64 = base64.b64encode(p.read_bytes()).decode()
        inner = f"<img src='data:image/png;base64,{b64}' style='height:34px;display:block;'>"
    else:
        inner = f"<span style='color:{color};font-weight:800;letter-spacing:1px;font-size:16px;'>{fallback}</span>"
    return (f"<div style='display:flex;flex-direction:column;align-items:center;gap:2px;'>"
            f"<span style='color:#9fb3cc;font-size:9px;letter-spacing:1.5px;'>{role}</span>"
            f"<div style='background:#fff;border-radius:7px;padding:6px 12px;"
            f"box-shadow:0 1px 4px rgba(0,0,0,.15);'>{inner}</div></div>")


def benford_chart(bt, key="benford"):
    """Interactive plotly chart if available; otherwise a built-in fallback."""
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_bar(x=bt["digit"], y=bt["observed"], name="Observed", marker_color="#2E5E8C")
        fig.add_scatter(x=bt["digit"], y=bt["expected"], name="Benford expected",
                        mode="lines+markers", line=dict(color=JK_RED, width=3))
        fig.update_layout(height=250, margin=dict(t=8, b=8), xaxis_title="Leading digit",
                          yaxis_title="Proportion", legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True, key=key)
    except ModuleNotFoundError:
        chart_df = bt.set_index("digit")[["observed", "expected"]]
        st.bar_chart(chart_df, height=250, key=key)
        st.caption("Tip: `pip install plotly` for the richer interactive chart.")


def card(col, label, value, color=NAVY, sub=""):
    col.markdown(
        f"<div style='background:#fff;border:1px solid #e6ecf4;border-left:5px solid {color};"
        f"border-radius:10px;padding:12px 16px;box-shadow:0 1px 4px rgba(31,58,95,.06);'>"
        f"<div style='font-size:12px;color:#6b7a90;font-weight:600'>{label}</div>"
        f"<div style='font-size:25px;font-weight:800;color:{color};line-height:1.2'>{value}</div>"
        f"<div style='font-size:11px;color:#90a0b5'>{sub}</div></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------- header
st.markdown(f"""
<style>
  .block-container {{padding-top:1.1rem;}}
  .gst-accent {{height:5px;border-radius:4px;margin-bottom:8px;
    background:linear-gradient(90deg,{JK_RED} 0%,{JK_RED} 50%,{IBM_BLUE} 50%,{IBM_BLUE} 100%);}}
  .gst-hero {{background:linear-gradient(135deg,#16263C 0%,#1F3A5F 38%,#2E5E8C 100%);
    border-radius:12px;padding:16px 24px;display:flex;align-items:center;
    justify-content:space-between;box-shadow:0 2px 12px rgba(31,58,95,.28);}}
  .gst-hero h1 {{color:#fff;margin:0;font-size:22px;font-weight:800;}}
  .gst-hero p {{color:#d6e3f2;margin:3px 0 0;font-size:12.5px;}}
  .gst-logos {{display:flex;gap:14px;align-items:center;}}
  .stTabs [data-baseweb="tab-list"] {{gap:6px;}}
  .stTabs [data-baseweb="tab"] {{font-weight:600;}}
</style>
<div class='gst-accent'></div>
<div class='gst-hero'>
  <div><h1>GST Anomaly &amp; Fraud Analytics Platform</h1>
  <p>Supply Chain Accounts Payable &middot; ITC Mismatch &middot; Fake-Invoice &middot; Tax-Evasion Detection</p></div>
  <div class='gst-logos'>{_logo("jkcl_logo.png","JK CEMENT",JK_RED,"CLIENT")}{_logo("ibm_logo.png","IBM",IBM_BLUE,"DELIVERED BY")}</div>
</div>
""", unsafe_allow_html=True)
st.caption("⚠️  Flags are risk indicators for review — not fraud verdicts. "
           "Every flagged item needs human confirmation before any vendor action.")

try:
    ref, model = load_assets()
except Exception as e:
    st.error(f"Could not load reference masters / model. Check ./reference and "
             f"the paths at the top of app.py.\n\n{e}")
    st.stop()

tab_dash, tab_scan, tab_chat, tab_single, tab_help = st.tabs(
    ["📊 Dashboard", "🛡️ Batch Scan & Validate", "💬 Ask the GST Analyst",
     "🔍 Single-invoice check", "ℹ️ How to use"])


# ===========================================================================
# Shared renderers
# ===========================================================================
def render_recon(res, dashboard=False):
    ctx = "dash" if dashboard else "scan"
    c = st.columns(4)
    card(c[0], "Records", f"{res['rows']:,}")
    card(c[1], "Total ITC at Risk", inr(res["total_itc_at_risk"]), JK_RED)
    card(c[2], "Fake-invoice candidates", f"{res['fake_candidates']:,}",
         "#C0392B", inr(res["fake_itc"]) + " ITC")
    chk_fail = int(res["checklist_summary"]["Fail"].sum())
    card(c[3], "Checklist failures", f"{chk_fail:,}", "#E67E22")

    st.markdown(f"> {res['narrative']}")

    a, b = st.columns(2)
    with a:
        st.markdown("**ITC at risk by mismatch type**")
        st.dataframe(res["by_type"].rename(columns={"ITC_at_Risk": "ITC at Risk"}),
                     hide_index=True, use_container_width=True, key=f"bytype_{ctx}",
                     column_config={"ITC at Risk": st.column_config.NumberColumn(format="%.0f")})
    with b:
        st.markdown("**Top suppliers by ITC at risk**")
        st.dataframe(res["top_vendors"].rename(columns={"ITC_at_Risk": "ITC at Risk"}).head(8),
                     hide_index=True, use_container_width=True, key=f"topvend_{ctx}",
                     column_config={"ITC at Risk": st.column_config.NumberColumn(format="%.0f")})

    # forensic strip
    st.markdown("**Forensic analytics**")
    f = st.columns(5)
    bt, mad, verdict = res["benford"]
    vcol = "#C0392B" if "Nonconformity" in verdict else ("#E67E22" if "Marginal" in verdict else "#27AE60")
    card(f[0], "Benford MAD", f"{mad:.4f}", vcol, verdict)
    card(f[1], "Invalid GSTINs", f"{int(res['clusters'].iloc[0]['Structurally invalid GSTINs']):,}", "#C0392B")
    card(f[2], "Shell-network rows", f"{res['high_network']:,}", "#E67E22")
    card(f[3], "Duplicates", f"{res['n_exact_dup']:,}", "#E67E22", f"{res['n_split']:,} split")
    card(f[4], "Round / threshold", f"{res['n_round']:,}", NAVY, f"{res['n_threshold']:,} just-under")

    benford_chart(bt, key=f"benford_{ctx}")

    if not dashboard:
        st.markdown("##### Validation queue — every flagged invoice, ranked")
        view = res["combined"]
        bands = st.multiselect("Risk filter", ["HIGH", "MEDIUM", "LOW"],
                               default=["HIGH", "MEDIUM"], key=f"reconfilter_{ctx}")
        view = view[view["Risk"].isin(bands)].copy()
        view["Decision"] = "Pending"
        view["Remarks"] = ""
        edited = st.data_editor(
            view, hide_index=True, use_container_width=True, height=420,
            key=f"recedit_{ctx}",
            column_config={
                "ITC at Risk": st.column_config.NumberColumn(format="%.0f"),
                "Fraud Score": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                "Decision": st.column_config.SelectboxColumn(
                    options=["Pending", "Accept", "Reverse ITC", "Hold", "Follow up supplier", "Query"]),
            },
            disabled=[col for col in view.columns if col not in ("Decision", "Remarks")])
        st.download_button("⬇️ Export validated queue", to_excel_bytes(edited),
                           file_name="gst_validated_queue.xlsx", key=f"recexp_{ctx}")


def render_posted(res, dashboard=False):
    ctx = "dash" if dashboard else "scan"
    c = st.columns(4)
    card(c[0], "Invoices", f"{res['rows']:,}")
    card(c[1], "🔴 HIGH", f"{res['high']:,}", "#C0392B")
    card(c[2], "🟠 MEDIUM", f"{res['medium']:,}", "#E67E22")
    card(c[3], "🟢 LOW", f"{res['low']:,}", "#27AE60")

    st.markdown("**Forensic analytics**")
    f = st.columns(4)
    bt, mad, verdict = res["benford"]
    vcol = "#C0392B" if "Nonconformity" in verdict else ("#E67E22" if "Marginal" in verdict else "#27AE60")
    card(f[0], "Benford MAD", f"{mad:.4f}", vcol, verdict)
    card(f[1], "Duplicates", f"{res['n_exact_dup']:,}", "#E67E22", f"{res['n_split']:,} split")
    card(f[2], "Round amounts", f"{res['n_round']:,}", NAVY)
    card(f[3], "Just-under-threshold", f"{res['n_threshold']:,}", NAVY)

    benford_chart(bt, key=f"benfordp_{ctx}")

    if not dashboard:
        st.markdown("##### Validation queue — ranked by risk")
        view = res["combined"]
        bands = st.multiselect("Risk filter", ["HIGH", "MEDIUM", "LOW"],
                               default=["HIGH", "MEDIUM"], key=f"postfilter_{ctx}")
        view = view[view["Band"].isin(bands)].copy()
        view["Decision"] = "Pending"
        view["Remarks"] = ""
        edited = st.data_editor(
            view, hide_index=True, use_container_width=True, height=420,
            key=f"postedit_{ctx}",
            column_config={
                "Risk Score": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                "Amount": st.column_config.NumberColumn(format="%.0f"),
                "Decision": st.column_config.SelectboxColumn(
                    options=["Pending", "Validated", "Hold", "Query"]),
            },
            disabled=[col for col in view.columns if col not in ("Decision", "Remarks")])
        st.download_button("⬇️ Export validated queue", to_excel_bytes(edited),
                           file_name="gst_validated_queue.xlsx", key=f"postexp_{ctx}")


# ===========================================================================
# TAB — Batch Scan & Validate (the one smart engine)
# ===========================================================================
with tab_scan:
    st.markdown("Upload **any** GST / invoice file. The platform auto-detects the "
                "type and runs **everything applicable** — ML scoring or 2B-PR "
                "reconciliation, the 12-point checklist, GSTIN validation, Benford, "
                "vendor-network, duplicate & threshold analysis, and fraud scoring.")
    up = st.file_uploader("Upload file (.xlsx / .xlsm / .xlsb / .csv)",
                          type=["xlsx", "xlsm", "xlsb", "csv"], key="scan")
    if up is not None:
        from gst_ingest import read_any, detect_file_type
        sig = f"{up.name}-{up.size}"
        if st.session_state.get("analysis_sig") != sig:
            with st.spinner("Reading & detecting file type …"):
                df, sheet = read_any(up, up.name)
                ftype, _ = detect_file_type(df)
            if ftype == "unknown":
                st.warning("Couldn't confidently classify the file. Choose how to treat it.")
                pick = st.radio("Treat as:", ["Posted-invoice data", "2B vs PR mismatch"],
                                horizontal=True, key="scanpick")
                ftype = "posted_invoice" if pick.startswith("Posted") else "recon"
            with st.spinner("Running full analysis (all techniques) …"):
                res = run_full_analysis(df, ftype, ref, score_frame)
            st.session_state["analysis"] = res
            st.session_state["analysis_name"] = up.name
            st.session_state["analysis_sig"] = sig
        res = st.session_state["analysis"]
        st.success(f"Analysed **{st.session_state['analysis_name']}** · "
                   f"detected **{res['kind']}** · {res['rows']:,} rows. "
                   "Summary now also on the **Dashboard** tab.")
        if res["kind"] == "recon":
            render_recon(res)
        else:
            render_posted(res)
    else:
        st.info("Upload a file to run the full validation + forensic analysis.")


# ===========================================================================
# TAB — Dashboard (last data used)
# ===========================================================================
with tab_dash:
    res = st.session_state.get("analysis")
    if res is None:
        st.info("No analysis yet. Upload a file in **Batch Scan & Validate** and "
                "this dashboard will summarise it.")
    else:
        st.markdown(f"#### Dashboard — *{st.session_state.get('analysis_name','last file')}*")
        if res["kind"] == "recon":
            render_recon(res, dashboard=True)
        else:
            render_posted(res, dashboard=True)


# ===========================================================================
# TAB — Ask the GST Analyst
# ===========================================================================
with tab_chat:
    from gst_assistant import narrative_summary, answer, llm_configured
    try:
        secrets = dict(st.secrets)
    except Exception:
        secrets = {}
    mode = "🤖 AI chat (LLM connected)" if llm_configured(secrets) else \
        "📊 Offline analyst (answers common questions; add an API key for free-form chat)"
    st.markdown(f"Ask about your data in plain English. **Mode: {mode}**")

    res = st.session_state.get("analysis")
    if res is None or res["kind"] != "recon":
        st.info("Analyse a **2B vs PR** file in Batch Scan & Validate first — then "
                "ask questions here about ITC at risk, fake invoices, vendors, etc.")
    else:
        f = res["findings"]
        a, b, c = st.columns(3)
        card(a, "Total ITC at Risk", inr(f["total_itc_at_risk"]), JK_RED)
        card(b, "Fake-invoice candidates", f"{f['fake_invoice_candidates']:,}", "#C0392B")
        card(c, "Candidate ITC", inr(f["fake_invoice_itc"]), "#C0392B")
        with st.expander("🕵️ Fake-invoice / evasion candidates (ranked)"):
            fr = f["fraud"].sort_values("Fraud Risk Score", ascending=False)
            fr = fr[fr["Risk Level"].isin(["HIGH", "MEDIUM"])]
            st.dataframe(fr[["Supplier", "Invoice No", "ITC at Risk", "Fraud Risk Score",
                             "Risk Level", "Indicators"]], hide_index=True,
                         use_container_width=True, height=280,
                         column_config={"ITC at Risk": st.column_config.NumberColumn(format="%.0f"),
                                        "Fraud Risk Score": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
        st.markdown("##### 💬 Chat")
        qp = st.columns(4)
        presets = ["Summarise the risks", "Top vendors by ITC at risk",
                   "How many fake invoices?", "Biggest process failures"]
        clicked = None
        for col, p in zip(qp, presets):
            if col.button(p, use_container_width=True):
                clicked = p
        for role, msg in st.session_state.get("chat_history", []):
            with st.chat_message(role):
                st.write(msg)
        typed = st.chat_input("Ask about ITC mismatches, fake invoices, vendors …")
        question = typed or clicked
        if question:
            st.session_state.setdefault("chat_history", []).append(("user", question))
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                with st.spinner("Analysing …"):
                    resp = answer(question, f, secrets)
                st.write(resp)
            st.session_state["chat_history"].append(("assistant", resp))


# ===========================================================================
# TAB — Single-invoice check
# ===========================================================================
with tab_single:
    st.markdown("##### Check one invoice on demand")
    with st.form("single"):
        a, b, c = st.columns(3)
        vendor = a.text_input("Vendor name")
        gstin = b.text_input("Vendor GSTIN")
        inv_no = c.text_input("Invoice number")
        d, e, f = st.columns(3)
        inv_date = d.date_input("Invoice date")
        post_date = e.date_input("Posting date")
        amount = f.number_input("Invoice amount (INR)", value=0.0, step=1000.0)
        g, h, i = st.columns(3)
        bplace = g.text_input("Business place (e.g. 9007)")
        tcode = h.text_input("Tax code (e.g. GD)")
        dtype = i.text_input("Doc type (e.g. RE)")
        submitted = st.form_submit_button("Validate invoice")
    if submitted:
        row = pd.DataFrame([{
            "VendorName": vendor, "vendor_gstin": gstin, "Reference": inv_no,
            "DocDate": pd.Timestamp(inv_date), "PstngDate": pd.Timestamp(post_date),
            "DocAmt": amount, "business_place": bplace, "tax_code": tcode,
            "Type": dtype, "Account": gstin or vendor, "DocumentNo": 0}])
        res = score_frame(row).iloc[0]
        flags = int(res.get("rule_flag_count", 0))
        if flags >= 2:
            band, basis = "HIGH", f"{flags} rule violations"
        elif flags == 1:
            band, basis = "MEDIUM", "1 rule violation"
        else:
            band, basis = res["risk_band"], "ML pattern score (no rule flag)"
        st.markdown(f"<div style='background:{BAND_COLOR.get(band,'#888')};color:#fff;"
                    f"padding:12px 16px;border-radius:8px;font-size:18px;font-weight:700'>"
                    f"{band} RISK · {basis}</div>", unsafe_allow_html=True)
        st.write("**Rule checks:**", res.get("rule_reasons") or "No deterministic rule flag")
        if gstin.strip():
            from gstin_validator import validate as gvalidate
            gv = gvalidate(gstin)
            if gv["valid"]:
                st.success(f"GSTIN valid ✓ — {gv['state']} · PAN {gv['pan']}"
                           + (" · JK Cement entity" if gv["is_jk"] else ""))
            else:
                st.error(f"GSTIN check failed ✗ — {gv['reason']} (possible fake/typo)")
        if dtype.strip():
            from gst_process import DOCTYPE_MASTER
            dt = dtype.strip().upper()
            if dt in DOCTYPE_MASTER:
                m = DOCTYPE_MASTER[dt]
                st.info(f"Doc type **{dt}** — {m['desc']} · charge {m['charge']} · {m['cat'].title()}")
            else:
                st.warning(f"Doc type '{dt}' is not in the SOP §4 master — confirm with the team.")
        st.caption("On a single invoice the rule + GSTIN checks are the primary "
                   "signal; the ML score needs vendor history to be reliable.")


# ===========================================================================
# TAB — How to use
# ===========================================================================
with tab_help:
    st.markdown("""
**Batch Scan & Validate** — the one engine. Upload any GST/invoice file; it
auto-detects the type and runs everything applicable:
- *Posted-invoice data* → ML risk scoring + rules + Benford + duplicates + Isolation Forest
- *2B-vs-PR mismatch* → ITC-at-risk reconciliation + **16-point SOP checklist** + GSTIN
  validation + Benford + vendor-network/shell signals + duplicates + fraud scoring

The 16-point checklist matches JK Cement's documented SOP, including **document-type
validation** (SOP §4: KR/RE/RC/US… with RCM/FCM consistency), **GST-rate validation**
(ZGSTR2N), **Inter/Intra**, **eligibility**, and a **Suggested SAP correction action**
per invoice (FB60→F-02 / MIRO→ZFI_AP). Set a **Decision** per row and export.

**Dashboard** — a summary of the **last file** you analysed (KPIs, mismatch mix,
top vendors, forensic strip, Benford chart).

**Ask the GST Analyst** — plain-English Q&A on the last analysed file, with a
ranked fake-invoice / evasion candidate list. Add an LLM key in
`.streamlit/secrets.toml` for free-form chat; works offline otherwise.

**Single-invoice check** — rule + GSTIN validation for one invoice on demand.

Branding: drop `assets/jkcl_logo.png` and `assets/ibm_logo.png` to replace the
text wordmarks. Theme colours live in `.streamlit/config.toml`.

*All flags are risk indicators for review — not fraud verdicts.*
""")
