"""
GST Anomaly Detection Tool - GSTIN Validator
JK Cement | Supply Chain Accounts Payable

Two layers:
  1. OFFLINE  - structural + check-digit validation (mod-36 algorithm). Catches
     fabricated / mistyped GSTINs instantly, no internet, no cost.
  2. LIVE     - optional adapter to a GST Suvidha Provider (GSP) / third-party
     verification API to confirm a GSTIN is registered & ACTIVE and read the
     legal name. You supply the endpoint + API key; structure stays the same.

A GSTIN is 15 chars: 2 state code + 10 PAN + 1 entity + 'Z' + 1 check digit.
"""

import re
import pandas as pd

CP = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MOD = 36
JK_PAN = "AABCJ0355R"   # JK Cement PAN embedded in every recipient GSTIN

STATE_CODES = {
    "01": "Jammu & Kashmir", "02": "Himachal Pradesh", "03": "Punjab",
    "04": "Chandigarh", "05": "Uttarakhand", "06": "Haryana", "07": "Delhi",
    "08": "Rajasthan", "09": "Uttar Pradesh", "10": "Bihar", "11": "Sikkim",
    "12": "Arunachal Pradesh", "13": "Nagaland", "14": "Manipur", "15": "Mizoram",
    "16": "Tripura", "17": "Meghalaya", "18": "Assam", "19": "West Bengal",
    "20": "Jharkhand", "21": "Odisha", "22": "Chhattisgarh", "23": "Madhya Pradesh",
    "24": "Gujarat", "25": "Daman & Diu", "26": "Dadra & Nagar Haveli",
    "27": "Maharashtra", "28": "Andhra Pradesh (old)", "29": "Karnataka",
    "30": "Goa", "31": "Lakshadweep", "32": "Kerala", "33": "Tamil Nadu",
    "34": "Puducherry", "35": "Andaman & Nicobar", "36": "Telangana",
    "37": "Andhra Pradesh", "38": "Ladakh", "97": "Other Territory",
}

_GSTIN_RE = re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][0-9A-Z]Z[0-9A-Z]$")


def check_digit(g14):
    factor, total = 2, 0
    for ch in reversed(g14):
        a = factor * CP.index(ch)
        factor = 1 if factor == 2 else 2
        total += (a // MOD) + (a % MOD)
    return CP[(MOD - (total % MOD)) % MOD]


def validate(gstin):
    """Offline validation. Returns dict with valid flag + reason + parts."""
    g = str(gstin).strip().upper().replace("`", "")
    res = {"gstin": g, "valid": False, "reason": "", "state_code": "",
           "state": "", "pan": "", "is_jk": False}
    if len(g) != 15:
        res["reason"] = "Length not 15"; return res
    if not all(c in CP for c in g):
        res["reason"] = "Invalid characters"; return res
    if not _GSTIN_RE.match(g):
        res["reason"] = "Format mismatch"; return res
    res["state_code"] = g[:2]
    res["state"] = STATE_CODES.get(g[:2], "Unknown state code")
    res["pan"] = g[2:12]
    res["is_jk"] = (res["pan"] == JK_PAN)
    if g[:2] not in STATE_CODES:
        res["reason"] = "Unknown state code"; return res
    if check_digit(g[:14]) != g[14]:
        res["reason"] = "Check-digit failed (possible fake/typo)"; return res
    res["valid"] = True
    res["reason"] = "OK"
    return res


def validate_series(gstins):
    """Vectorised-ish helper: returns DataFrame of validation results."""
    return pd.DataFrame([validate(g) for g in gstins])


# ---------------------------------------------------------------------------
# LIVE adapter (optional) — plug in your GSP / third-party verification API.
# Left as a clearly-marked stub: no calls are made unless you provide creds.
# ---------------------------------------------------------------------------
def verify_live(gstin, api_url=None, api_key=None, timeout=8):
    """
    Confirm a GSTIN is registered & active via an external API.

    To enable: pass your provider's `api_url` + `api_key`. Typical providers
    return JSON with the trade/legal name and registration status. Adapt the
    response parsing to your provider's schema where marked below.

    Returns: dict(active: bool|None, legal_name: str|None, status: str)
    """
    if not api_url or not api_key:
        return {"active": None, "legal_name": None,
                "status": "LIVE check not configured (offline validation only)"}
    try:
        import requests
        r = requests.get(api_url.format(gstin=gstin),
                         headers={"Authorization": f"Bearer {api_key}"},
                         timeout=timeout)
        r.raise_for_status()
        data = r.json()
        # ---- adapt these two lines to your provider's JSON schema ----
        legal_name = data.get("legal_name") or data.get("lgnm")
        status = data.get("status") or data.get("sts")
        # --------------------------------------------------------------
        return {"active": str(status).upper() in ("ACTIVE", "ACT"),
                "legal_name": legal_name, "status": status or "unknown"}
    except Exception as e:
        return {"active": None, "legal_name": None, "status": f"LIVE error: {e}"}
