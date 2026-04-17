"""
Loan grade UI — calls the FastAPI service over HTTP.

Run the API first (from repo root so artifacts resolve):
  uvicorn predict:app --host 0.0.0.0 --port 9696

Then run this app:
  streamlit run streamlit_app.py

Optional env:
  LOAN_API_BASE=http://127.0.0.1:9696   (default)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, cast

import httpx
import pandas as pd
import streamlit as st

from explain import _display_value

DEFAULT_API_BASE = "http://127.0.0.1:9696"
REQUEST_TIMEOUT_S = 120.0


def _api_base() -> str:
    return os.environ.get("LOAN_API_BASE", DEFAULT_API_BASE).rstrip("/")


def _num(name: str, label: str, default: float, step: float = 1.0) -> float:
    v = st.number_input(label, min_value=0.0, value=float(default), step=step, key=name)
    return cast(float, v)


def _maybe_cat(label: str, options: List[str], key: str) -> Optional[str]:
    choice = st.selectbox(label, options, key=key)
    if choice == "":
        return None
    return choice


def _build_payload(
    loan_amnt: float,
    annual_inc: float,
    dti: float,
    revol_util: float,
    delinq_2yrs: float,
    pub_rec: float,
    pub_rec_bankruptcies: float,
    num_tl_90g_dpd_24m: float,
    pct_tl_nvr_dlq: float,
    installment: float,
    funded_amnt: float,
    bc_util: float,
    term: Optional[str],
    home_ownership: Optional[str],
    application_type: Optional[str],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "loan_amnt": loan_amnt,
        "annual_inc": annual_inc,
        "dti": dti,
        "revol_util": revol_util,
        "delinq_2yrs": delinq_2yrs,
        "pub_rec": pub_rec,
        "pub_rec_bankruptcies": pub_rec_bankruptcies,
        "num_tl_90g_dpd_24m": num_tl_90g_dpd_24m,
        "pct_tl_nvr_dlq": pct_tl_nvr_dlq,
        "installment": installment,
        "funded_amnt": funded_amnt,
        "bc_util": bc_util,
    }
    if term:
        out["term"] = term
    if home_ownership:
        out["home_ownership"] = home_ownership
    if application_type:
        out["application_type"] = application_type
    return out


def _payload_looks_high_risk(payload: Dict[str, Any]) -> bool:
    """Heuristic: fields that usually imply stress vs safest grade buckets."""

    def _f(key: str) -> Optional[float]:
        v = payload.get(key)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    dti = _f("dti")
    rev_u = _f("revol_util")
    bc_u = _f("bc_util")
    dq = _f("delinq_2yrs")
    br = _f("pub_rec_bankruptcies")
    dpd = _f("num_tl_90g_dpd_24m")
    return bool(
        (dti is not None and dti >= 40)
        or (rev_u is not None and rev_u >= 90)
        or (bc_u is not None and bc_u >= 90)
        or (dq is not None and dq >= 1)
        or (br is not None and br >= 1)
        or (dpd is not None and dpd >= 3)
    )


def _explanation_value_str(v: Any) -> str:
    """Single string dtype for the explanation table (mixed raw numerics + categoricals break PyArrow)."""
    if v is None:
        return "—"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return str(v)
    return str(v)


def _reliability_warnings(
    probs: Dict[str, Any], payload: Dict[str, Any], grade: str
) -> List[str]:
    """Surface OOD / pathological-softmax cases so users do not trust absurd inputs blindly."""
    out: List[str] = []

    def _f(key: str) -> Optional[float]:
        v = payload.get(key)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    la = payload.get("loan_amnt")
    fa = payload.get("funded_amnt")
    try:
        la_f = float(la) if la is not None else None
        fa_f = float(fa) if fa is not None else None
    except (TypeError, ValueError):
        la_f = fa_f = None

    # Human-obvious stress vs model saying safest buckets — the net is wrong here, not "low risk".
    if str(grade).upper() in ("A", "B") and _payload_looks_high_risk(payload):
        out.append(
            "**This profile does not look low-risk to a person** (high DTI and/or utilization, "
            "delinquencies, bankruptcies, or serious DPD)—yet the model predicted a safer letter. "
            "That is a **model failure / OOD artifact**, not a statement that you are actually low risk. "
            "The model behind this app is not validated for extreme or inconsistent combinations of fields."
        )

    # Historical LC-style personal loans in public datasets are often ~$1k–$40k; larger is OOD.
    ood_amt = 50_000.0
    if la_f is not None and la_f >= ood_amt:
        out.append(
            f"**Loan amount ({la_f:,.0f})** is far above the range most rows in the training data "
            f"were built from. The MLP can behave strangely out-of-distribution (e.g. **counterintuitive "
            f"grades or ~100% on one letter**). Treat this prediction as a demo artifact, not a real "
            f"risk assessment."
        )
    if fa_f is not None and fa_f >= ood_amt:
        out.append(
            f"**Funded amount ({fa_f:,.0f})** is also very high vs typical training values—same OOD "
            "caveat applies."
        )

    if probs:
        mx = max(float(v) for v in probs.values() if isinstance(v, (int, float)))
        if mx >= 0.995:
            out.append(
                "**Very peaked probabilities** (≈100% on one grade) often mean the input sits where "
                "the network extrapolates badly—not that the model is objectively certain. Try smaller "
                "loan amounts and realistic funded amounts to see more balanced `grade_probabilities`."
            )

    if la_f is not None and fa_f is not None and la_f > 0 and fa_f > 0:
        if la_f > 5 * fa_f or fa_f > 5 * la_f:
            out.append(
                "**Loan amount and funded amount are very different.** In real listings they are usually "
                "close; this mismatch can land in a weird part of the feature space for this capstone model."
            )
        elif la_f > 2 * fa_f or fa_f > 2 * la_f:
            out.append(
                "**Loan amount is much larger than funded amount (or the reverse).** That is unusual for "
                "a single listing and pushes the model off the training manifold—grades can be meaningless."
            )

    return out


def main() -> None:
    st.set_page_config(page_title="Loan grade predictor", layout="wide")
    st.title("Loan application — grade and explanation")
    st.info(
        "**Who this is for:** Educators, students, and data practitioners exploring a **capstone-style** "
        "risk model—not borrowers applying for real credit, and **not** Lending Club or any lender’s "
        "official system.\n\n"
        "**What it does:** Sends credit-style fields to a demo API that predicts a historical **grade "
        "bucket (A–G)** and returns technical SHAP-style drivers (feature names and values come from the "
        "model pipeline, not consumer-friendly disclosures).\n\n"
        "**What it is not:** Legal, financial, or underwriting advice; a replacement for real compliance, "
        "fair-lending review, or human decisions."
    )
    st.caption(
        f"API: `{_api_base()}` — start with `uvicorn predict:app --host 0.0.0.0 --port 9696` from the project root."
    )

    with st.sidebar:
        st.markdown(
            "**Audience:** Demo / learning / analyst-style use only—not production borrower-facing."
        )
        st.header("Explanation params")
        top_n = st.number_input("top_n", min_value=1, max_value=50, value=10, step=1)
        max_evals = st.number_input("max_evals", min_value=50, max_value=2000, value=256, step=1)
        st.link_button("Open API docs", f"{_api_base()}/docs")

    # enter_to_submit=False: only the submit button submits (not Enter in number inputs).
    with st.form("loan_form", enter_to_submit=False):
        st.subheader("Application (curated fields)")
        c1, c2, c3 = st.columns(3)
        with c1:
            loan_amnt = _num("loan_amnt", "Loan amount", 10_000.0, 500.0)
            annual_inc = _num("annual_inc", "Annual income", 75_000.0, 1000.0)
            dti = _num("dti", "DTI (%)", 15.0, 0.5)
            revol_util = _num("revol_util", "Revolving utilization (%)", 35.0, 1.0)
        with c2:
            delinq_2yrs = _num("delinq_2yrs", "Delinquencies (2y)", 0.0, 1.0)
            pub_rec = _num("pub_rec", "Public records", 0.0, 1.0)
            pub_rec_bankruptcies = _num("pub_rec_bankruptcies", "Public record bankruptcies", 0.0, 1.0)
            num_tl_90g_dpd_24m = _num("num_tl_90g_dpd_24m", "90+ DPD tradelines (24m)", 0.0, 1.0)
        with c3:
            pct_tl_nvr_dlq = _num("pct_tl_nvr_dlq", "% tradelines never delinq", 95.0, 1.0)
            installment = _num("installment", "Installment payment", 350.0, 10.0)
            funded_amnt = _num("funded_amnt", "Funded amount", 10_000.0, 500.0)
            bc_util = _num("bc_util", "BC utilization (%)", 40.0, 1.0)

        st.subheader("Categorical (optional)")
        cc1, cc2 = st.columns(2)
        with cc1:
            term = _maybe_cat(
                "Term",
                ["", " 36 months", " 60 months"],
                "term",
            )
            home_ownership = _maybe_cat(
                "Home ownership",
                ["", "RENT", "MORTGAGE", "OWN", "ANY", "NONE", "OTHER"],
                "home_ownership",
            )
        with cc2:
            application_type = _maybe_cat(
                "Application type",
                ["", "Individual", "Joint App"],
                "application_type",
            )

        st.divider()
        submitted = st.form_submit_button(
            "Submit application — get grade and explanation",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return

    payload = _build_payload(
        loan_amnt=loan_amnt,
        annual_inc=annual_inc,
        dti=dti,
        revol_util=revol_util,
        delinq_2yrs=delinq_2yrs,
        pub_rec=pub_rec,
        pub_rec_bankruptcies=pub_rec_bankruptcies,
        num_tl_90g_dpd_24m=num_tl_90g_dpd_24m,
        pct_tl_nvr_dlq=pct_tl_nvr_dlq,
        installment=installment,
        funded_amnt=funded_amnt,
        bc_util=bc_util,
        term=term,
        home_ownership=home_ownership,
        application_type=application_type,
    )

    url = f"{_api_base()}/explain"
    params = {"top_n": int(top_n), "max_evals": int(max_evals)}

    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT_S) as client:
            r = client.post(url, params=params, json=payload)
    except httpx.ConnectError as e:
        st.error(f"Could not connect to API at `{_api_base()}`. Is uvicorn running? ({e})")
        return
    except httpx.TimeoutException:
        st.error("Request timed out. Try lowering max_evals or top_n, or retry.")
        return

    if r.status_code != 200:
        st.error(f"API returned {r.status_code}: {r.text[:2000]}")
        return

    data = r.json()
    grade = data.get("grade", "?")
    risk = data.get("risk_level", "?")
    explanation = data.get("explanation", [])

    st.success("Response received")
    probs = data.get("grade_probabilities") or {}
    for msg in _reliability_warnings(probs, payload, str(grade)):
        st.warning(msg)

    risk_contradicts_input = str(grade).upper() in ("A", "B") and _payload_looks_high_risk(payload)
    risk_unreliable_msg = (
        "Unreliable — not low risk for this input. See the yellow warnings above; the API may still "
        "return a friendly risk label that does not match these fields."
    )

    st.info(
        "The model uses **all** fields together. Changing only income often does **not** change the "
        "letter grade: other features (DTI, utilization, delinquencies, etc.) can dominate. "
        "The explanation table shows only the **top N drivers by |SHAP|** for this prediction, so "
        "`annual_inc` may not appear there even when it was sent to the API."
    )

    m1, m2, m3 = st.columns([1.15, 2.5, 1.15])
    with m1:
        st.metric("Predicted grade", grade)
    with m2:
        # st.metric truncates long values; markdown wraps within the wider column.
        if risk_contradicts_input:
            st.caption("Risk level")
            st.markdown(risk_unreliable_msg)
        else:
            st.metric("Risk level", risk)
    with m3:
        p_hat = probs.get(str(grade))
        st.metric(
            "Probability of predicted grade",
            f"{p_hat:.2%}" if isinstance(p_hat, (int, float)) else "—",
        )

    if probs:
        st.subheader("Probability over all grades (A–G)")
        prob_df = pd.DataFrame(
            sorted(probs.items(), key=lambda kv: kv[0]),
            columns=["Grade", "Probability"],
        ).set_index("Grade")
        st.bar_chart(prob_df)
        with st.expander("Grade probabilities (JSON)"):
            st.json(probs)

    with st.expander("Request sent to API (verify income and other values)"):
        st.json(payload)

    if explanation:
        st.subheader("Explanation (top drivers)")
        st.caption(
            "**Value** is recomputed here from **your form payload** (same rules as the API): "
            "only fields you actually sent appear; **—** means that row’s feature was not in your "
            "submission (the model may still use defaults internally). **Restart uvicorn** after "
            "pulling backend changes so SHAP rows stay in sync."
        )
        # Re-apply _display_value from the request we sent so the table never shows stale scaled
        # numbers if an old API process is still running.
        rows_display = []
        for row in explanation:
            v = _display_value(row["feature"], payload)
            rows_display.append(
                {
                    "feature": row["feature"],
                    "value": _explanation_value_str(v),
                    "shap_value": row["shap_value"],
                    "direction": row["direction"],
                }
            )
        st.dataframe(
            pd.DataFrame(rows_display),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("No explanation rows returned.")


if __name__ == "__main__":
    main()
