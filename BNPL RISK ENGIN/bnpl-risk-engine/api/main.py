
# ============================================================
# BNPL RISK SCORING ENGINE
# STEP 6 - FASTAPI LIVE CREDIT RISK SCORING API
# ============================================================

from pathlib import Path
import logging

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ============================================================
# 1. LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BNPL_API")

# ============================================================
# 2. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "xgb_pd_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"

# ============================================================
# 3. VERIFY MODEL FILES
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"XGBoost model not found: {MODEL_PATH}"
    )

if not PREPROCESSOR_PATH.exists():
    raise FileNotFoundError(
        f"Preprocessor not found: {PREPROCESSOR_PATH}"
    )

# ============================================================
# 4. LOAD MODEL AND PREPROCESSOR
# ============================================================

logger.info("Loading XGBoost model...")

model = joblib.load(MODEL_PATH)

logger.info("Loading preprocessing pipeline...")

preprocessor = joblib.load(PREPROCESSOR_PATH)

logger.info("Model and preprocessor loaded successfully.")

# ============================================================
# 5. OPTIONAL SHAP EXPLAINER
# ============================================================

explainer = None
shap_available = False

try:
    import shap

    explainer = shap.TreeExplainer(model)
    shap_available = True

    logger.info("SHAP explainer initialized successfully.")

except Exception as error:
    logger.warning(
        "SHAP unavailable. Using model feature importance fallback: %s",
        error
    )

# ============================================================
# 6. FEATURE CONFIGURATION
# ============================================================

FEATURES = [
    "age",
    "monthly_income",
    "bureau_score",
    "existing_loans",
    "dti_ratio",
    "tenure_months",
    "order_value",
    "on_time_pct_hist",
    "device_tenure_days",
    "app_velocity_30d",
    "city_tier"
]

# ============================================================
# 7. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="BNPL Risk Scoring API",
    description=(
        "Real-time Buy Now Pay Later default probability "
        "and PDO credit scoring service."
    ),
    version="1.0.0"
)

# ============================================================
# 8. PDO SCORECARD CONFIGURATION
# ============================================================

BASE_SCORE = 600
BASE_ODDS = 50
PDO = 40

FACTOR = PDO / np.log(2)

OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)

# ============================================================
# 9. APPLICANT INPUT SCHEMA
# ============================================================

class Applicant(BaseModel):

    applicant_id: str | None = None

    age: int = Field(ge=18, le=75)

    monthly_income: float = Field(gt=0)

    bureau_score: float = Field(ge=300, le=900)

    existing_loans: int = Field(ge=0)

    dti_ratio: float = Field(ge=0, le=100)

    tenure_months: int = Field(ge=0)

    order_value: float = Field(gt=0)

    on_time_pct_hist: float = Field(ge=0, le=100)

    device_tenure_days: float = Field(ge=0)

    app_velocity_30d: int = Field(ge=0)

    city_tier: str

# ============================================================
# 10. RESPONSE SCHEMA
# ============================================================

class ScoreResponse(BaseModel):

    applicant_id_echo: str | None = None

    probability_of_default: float

    credit_score: int

    decision: str

    top_reason_codes: list[dict]

    explanation_method: str

# ============================================================
# 11. PDO CREDIT SCORE FUNCTION
# ============================================================

def pd_to_score(probability):

    probability = float(
        np.clip(probability, 0.000001, 0.999999)
    )

    odds = (1 - probability) / probability

    score = OFFSET + FACTOR * np.log(odds)

    score = np.clip(score, 300, 900)

    return int(round(float(score)))

# ============================================================
# 12. LOAN DECISION FUNCTION
# ============================================================

def get_decision(score):

    if score >= 700:
        return "APPROVE"

    elif score >= 600:
        return "APPROVE_LOW_LIMIT"

    elif score >= 500:
        return "MANUAL_REVIEW"

    return "DECLINE"

# ============================================================
# 13. FEATURE EXPLANATION FUNCTION
# ============================================================

def get_reason_codes(transformed_row):

    feature_names = list(
        preprocessor.get_feature_names_out()
    )

    # Try SHAP explanations first
    if shap_available and explainer is not None:

        try:
            shap_output = explainer.shap_values(
                transformed_row
            )

            if isinstance(shap_output, list):
                shap_output = shap_output[-1]

            values = np.asarray(shap_output)

            if values.ndim == 3:
                values = values[0, :, -1]

            elif values.ndim == 2:
                values = values[0]

            values = values.reshape(-1)

            if len(values) == len(feature_names):

                top_indices = np.argsort(
                    np.abs(values)
                )[::-1][:3]

                return [
                    {
                        "feature": str(feature_names[i]),
                        "shap_value": round(
                            float(values[i]), 4
                        ),
                        "effect": (
                            "increases_risk"
                            if values[i] > 0
                            else "reduces_risk"
                        )
                    }
                    for i in top_indices
                ], "SHAP"

        except Exception as error:
            logger.warning(
                "SHAP calculation failed: %s",
                error
            )

    # Fallback: global model feature importance
    importances = getattr(
        model, "feature_importances_", None
    )

    if importances is not None:

        importances = np.asarray(importances).reshape(-1)

        if len(importances) == len(feature_names):

            top_indices = np.argsort(
                importances
            )[::-1][:3]

            return [
                {
                    "feature": str(feature_names[i]),
                    "importance": round(
                        float(importances[i]), 4
                    ),
                    "effect": "global_importance_only"
                }
                for i in top_indices
            ], "MODEL_FEATURE_IMPORTANCE"

    return [], "NO_EXPLANATION_AVAILABLE"

# ============================================================
# 14. ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "project": "BNPL Risk Scoring Engine",
        "status": "running",
        "version": "1.0.0",
        "model": "XGBoost",
        "shap_available": shap_available
    }

# ============================================================
# 15. HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "shap_available": shap_available
    }

# ============================================================
# 16. LIVE CREDIT RISK SCORING
# ============================================================

@app.post("/score", response_model=ScoreResponse)
def score_applicant(applicant: Applicant):

    try:

        # Convert input into DataFrame
        applicant_data = pd.DataFrame(
            [[
                applicant.age,
                applicant.monthly_income,
                applicant.bureau_score,
                applicant.existing_loans,
                applicant.dti_ratio,
                applicant.tenure_months,
                applicant.order_value,
                applicant.on_time_pct_hist,
                applicant.device_tenure_days,
                applicant.app_velocity_30d,
                applicant.city_tier
            ]],
            columns=FEATURES
        )

        # Preprocess applicant
        transformed_data = preprocessor.transform(
            applicant_data
        )

        # Predict probability of default
        probability = float(
            model.predict_proba(
                transformed_data
            )[0, 1]
        )

        probability = float(
            np.clip(probability, 0, 1)
        )

        # Convert PD to credit score
        credit_score = pd_to_score(probability)

        # Loan decision
        decision = get_decision(credit_score)

        # Explain prediction
        reason_codes, explanation_method = get_reason_codes(
            transformed_data
        )

        return ScoreResponse(
            applicant_id_echo=applicant.applicant_id,
            probability_of_default=round(
                probability, 4
            ),
            credit_score=credit_score,
            decision=decision,
            top_reason_codes=reason_codes,
            explanation_method=explanation_method
        )

    except Exception as error:

        logger.exception("Applicant scoring failed")

        raise HTTPException(
            status_code=500,
            detail=f"Applicant scoring failed: {str(error)}"
        )

# ============================================================
# 17. END OF API
# ============================================================
