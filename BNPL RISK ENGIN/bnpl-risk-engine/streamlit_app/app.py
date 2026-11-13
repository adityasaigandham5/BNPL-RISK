
# ============================================================
# BNPL RISK SCORING ENGINE
# STEP 7 - STREAMLIT UNDERWRITER DASHBOARD
# ============================================================

from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BNPL Risk Intelligence",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# 2. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"

SCORED_DATA_PATH = DATA_DIR / "scored_test_set.csv"

API_URL = "http://127.0.0.1:8000"

# ============================================================
# 3. APPLICATION HEADER
# ============================================================

st.title("BNPL Risk Scoring Engine")
st.caption(
    "Credit Risk Intelligence | XGBoost | PDO Scorecard | FastAPI"
)

st.divider()

# ============================================================
# 4. LOAD PORTFOLIO DATA
# ============================================================

@st.cache_data
def load_portfolio():

    if not SCORED_DATA_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(SCORED_DATA_PATH)


scored = load_portfolio()

# ============================================================
# 5. CREATE DASHBOARD TABS
# ============================================================

tab1, tab2 = st.tabs([
    "Score an Applicant",
    "Portfolio Monitor"
])

# ============================================================
# TAB 1 - LIVE APPLICANT SCORING
# ============================================================

with tab1:

    st.subheader("Live BNPL Applicant Risk Assessment")

    st.write(
        "Enter applicant financial, credit and repayment information "
        "to generate a real-time risk score."
    )

    # Applicant inputs
    c1, c2, c3 = st.columns(3)

    with c1:

        applicant_id = st.text_input(
            "Applicant ID",
            value="BNPL001"
        )

        age = st.slider(
            "Age",
            min_value=18,
            max_value=75,
            value=28
        )

        monthly_income = st.number_input(
            "Monthly Income (INR)",
            min_value=1.0,
            max_value=1000000.0,
            value=45000.0,
            step=1000.0
        )

        bureau_score = st.slider(
            "Bureau Score",
            min_value=300,
            max_value=900,
            value=760
        )

    with c2:

        existing_loans = st.number_input(
            "Existing Loans",
            min_value=0,
            max_value=50,
            value=1
        )

        dti_ratio = st.slider(
            "Debt-to-Income Ratio (%)",
            min_value=0.0,
            max_value=100.0,
            value=25.0
        )

        tenure_months = st.number_input(
            "Account Tenure (Months)",
            min_value=0,
            max_value=600,
            value=12
        )

        order_value = st.number_input(
            "BNPL Order Value (INR)",
            min_value=1.0,
            max_value=500000.0,
            value=4999.0,
            step=500.0
        )

    with c3:

        on_time_pct_hist = st.slider(
            "Past On-Time Repayment (%)",
            min_value=0.0,
            max_value=100.0,
            value=95.0
        )

        device_tenure_days = st.number_input(
            "Device Tenure (Days)",
            min_value=0.0,
            max_value=10000.0,
            value=450.0
        )

        app_velocity_30d = st.number_input(
            "App Activity - Last 30 Days",
            min_value=0,
            max_value=10000,
            value=5
        )

        city_tier = st.selectbox(
            "City Tier",
            ["Tier1", "Tier2", "Tier3"]
        )

    st.divider()

    # ========================================================
    # SCORE APPLICANT BUTTON
    # ========================================================

    if st.button(
        "Calculate Credit Risk Score",
        type="primary",
        use_container_width=True
    ):

        payload = {
            "applicant_id": applicant_id,
            "age": age,
            "monthly_income": monthly_income,
            "bureau_score": bureau_score,
            "existing_loans": existing_loans,
            "dti_ratio": dti_ratio,
            "tenure_months": tenure_months,
            "order_value": order_value,
            "on_time_pct_hist": on_time_pct_hist,
            "device_tenure_days": device_tenure_days,
            "app_velocity_30d": app_velocity_30d,
            "city_tier": city_tier
        }

        try:

            with st.spinner("Calculating applicant risk..."):

                response = requests.post(
                    f"{API_URL}/score",
                    json=payload,
                    timeout=30
                )

            if response.status_code == 200:

                result = response.json()

                st.success("Applicant scoring completed!")

                st.subheader("Credit Risk Assessment")

                m1, m2, m3 = st.columns(3)

                pd_value = result["probability_of_default"]

                m1.metric(
                    "Probability of Default",
                    f"{pd_value * 100:.2f}%"
                )

                m2.metric(
                    "Credit Score",
                    f"{result['credit_score']}/900"
                )

                m3.metric(
                    "Decision",
                    result["decision"]
                )

                # Decision status
                decision = result["decision"]

                if decision == "APPROVE":
                    st.success("Applicant classified as APPROVE.")

                elif decision == "APPROVE_LOW_LIMIT":
                    st.info("Applicant classified as APPROVE - LOW LIMIT.")

                elif decision == "MANUAL_REVIEW":
                    st.warning("Applicant requires MANUAL REVIEW.")

                else:
                    st.error("Applicant classified as DECLINE.")

                st.subheader("Model Explanation")

                st.write(
                    "Explanation method:",
                    result.get(
                        "explanation_method",
                        "Not available"
                    )
                )

                reasons = result.get("top_reason_codes", [])

                if reasons:
                    st.dataframe(
                        pd.DataFrame(reasons),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No explanation details available.")

                with st.expander("View complete API response"):
                    st.json(result)

            else:

                st.error(
                    f"API returned HTTP {response.status_code}"
                )

                st.code(response.text)

        except requests.exceptions.ConnectionError:

            st.error(
                "Cannot connect to FastAPI. "
                "Start your API server on port 8000 first."
            )

        except requests.exceptions.Timeout:

            st.error(
                "API request timed out. Please try again."
            )

        except Exception as error:

            st.error(f"Scoring error: {error}")


# ============================================================
# TAB 2 - PORTFOLIO MONITOR
# ============================================================

with tab2:

    st.subheader("BNPL Portfolio Risk Monitoring")

    if scored.empty:

        st.warning(
            "Portfolio dataset not found. "
            "Run Step 5 and generate data/scored_test_set.csv."
        )

    else:

        required_columns = [
            "credit_score",
            "decision",
            "actual_default"
        ]

        missing = [
            col for col in required_columns
            if col not in scored.columns
        ]

        if missing:

            st.error(
                f"Missing required portfolio columns: {missing}"
            )

        else:

            # ================================================
            # KPI CARDS
            # ================================================

            total_applicants = len(scored)

            approve_rate = (
                scored["decision"].isin(
                    ["APPROVE", "APPROVE_LOW_LIMIT"]
                ).mean() * 100
            )

            decline_rate = (
                (scored["decision"] == "DECLINE").mean() * 100
            )

            portfolio_default_rate = (
                scored["actual_default"].mean() * 100
            )

            k1, k2, k3, k4 = st.columns(4)

            k1.metric(
                "Total Applicants",
                f"{total_applicants:,}"
            )

            k2.metric(
                "Approval Rate",
                f"{approve_rate:.2f}%"
            )

            k3.metric(
                "Decline Rate",
                f"{decline_rate:.2f}%"
            )

            k4.metric(
                "Portfolio Default Rate",
                f"{portfolio_default_rate:.2f}%"
            )

            st.divider()

            # ================================================
            # CREDIT SCORE DISTRIBUTION
            # ================================================

            st.subheader(
                "Credit Score Distribution by Decision Band"
            )

            fig1 = px.histogram(
                scored,
                x="credit_score",
                color="decision",
                nbins=50,
                title="Applicant Credit Score Distribution",
                labels={
                    "credit_score": "Credit Score",
                    "decision": "Loan Decision"
                },
                template="plotly_white"
            )

            st.plotly_chart(
                fig1,
                use_container_width=True
            )

            # ================================================
            # DEFAULT RATE BY DECISION
            # ================================================

            st.subheader(
                "Actual Default Rate by Decision Band"
            )

            default_rates = (
                scored.groupby("decision", as_index=False)
                ["actual_default"]
                .mean()
            )

            default_rates["default_rate_percent"] = (
                default_rates["actual_default"] * 100
            )

            fig2 = px.bar(
                default_rates,
                x="decision",
                y="default_rate_percent",
                color="decision",
                title="Actual Default Rate by Loan Decision",
                labels={
                    "decision": "Decision Band",
                    "default_rate_percent": "Default Rate (%)"
                },
                template="plotly_white"
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

            # ================================================
            # DECISION DISTRIBUTION
            # ================================================

            st.subheader("Loan Decision Distribution")

            decision_counts = (
                scored["decision"]
                .value_counts()
                .reset_index()
            )

            decision_counts.columns = [
                "decision",
                "applicants"
            ]

            fig3 = px.pie(
                decision_counts,
                names="decision",
                values="applicants",
                title="Applicant Decision Breakdown",
                hole=0.4
            )

            st.plotly_chart(
                fig3,
                use_container_width=True
            )

            # ================================================
            # PORTFOLIO DATA TABLE
            # ================================================

            st.subheader("Scored Applicant Records")

            st.dataframe(
                scored,
                use_container_width=True,
                hide_index=True
            )

            # ================================================
            # DOWNLOAD REPORT
            # ================================================

            csv_data = scored.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Portfolio Report",
                data=csv_data,
                file_name="bnpl_portfolio_report.csv",
                mime="text/csv",
                use_container_width=True
            )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "BNPL Risk Scoring Engine | XGBoost + PDO Scorecard + "
    "FastAPI + Streamlit | Development Demo"
)
