"""
esg_model.py
=============

Training, prediction, and explainability layer for the ESG Risk Predictor.

The module exposes a single high-level class, :class:`ESGRiskModel`, which:

1. Generates a labelled synthetic training set whose ground-truth risk
   scores follow a transparent, hand-crafted formula.  Because the
   formula is known we can sanity-check the model AND the SHAP
   explanations.

2. Fits one Random Forest regressor *per pillar* (E, S, G).  Three
   single-output models give us cleaner SHAP attributions than one
   multi-output model.

3. Predicts E, S, G and overall ESG risk for a new organisation, and
   returns a SHAP breakdown describing **why** the model produced that
   number.

The module is intentionally model-agnostic at the boundary - swap the
RandomForestRegressor for any other tree model and everything else
keeps working because we use ``shap.TreeExplainer``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from data_sources import (
    COUNTRY_INDICATORS,
    FEATURE_COLUMNS,
    INDUSTRY_BENCHMARKS,
    classify_risk,
    generate_company_features,
)


# Pillar weights used to roll up E, S, G into the overall ESG risk.
# Equal weights are the default in most major frameworks (GRI, SASB).
PILLAR_WEIGHTS = {"E": 1 / 3, "S": 1 / 3, "G": 1 / 3}


# ---------------------------------------------------------------------------
# Result containers - dataclasses keep the API self-documenting.
# ---------------------------------------------------------------------------

@dataclass
class PillarPrediction:
    """Prediction + explanation for a single pillar (E, S, or G)."""
    score: float                       # Predicted risk score (0-100)
    label: str                         # "Low" / "Medium" / "High"
    base_value: float                  # Model's expected output
    shap_values: Dict[str, float]      # Per-feature SHAP contribution


@dataclass
class ESGPrediction:
    """Aggregated prediction across all three pillars + overall ESG risk."""
    environmental: PillarPrediction
    social: PillarPrediction
    governance: PillarPrediction
    overall_score: float
    overall_label: str
    feature_row: pd.DataFrame          # The features that produced the result


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

def _synthesize_training_data(n_samples: int = 1500, seed: int = 42) -> pd.DataFrame:
    """
    Build a labelled training set whose target values follow a known
    deterministic-plus-noise formula.

    Why synthetic data?
        Real ESG ratings are licensed and expensive.  For a demo we want
        a realistic *shape* so the model can learn something meaningful
        - without claiming to predict real-world ratings.

    The targets (e_risk, s_risk, g_risk) are deliberately built from
    the same features that will be available at inference time, but
    with non-linear interactions and Gaussian noise so the model has
    something to learn instead of memorising.
    """
    rng = np.random.default_rng(seed)
    countries = list(COUNTRY_INDICATORS.keys())
    industries = list(INDUSTRY_BENCHMARKS.keys())

    rows: List[pd.Series] = []
    for _ in range(n_samples):
        country = rng.choice(countries)
        industry = rng.choice(industries)
        c = COUNTRY_INDICATORS[country]
        ind = INDUSTRY_BENCHMARKS[industry]

        # Random per-company traits (independent of org name during training).
        company_size = rng.beta(2, 5) * 100
        years_in_operation = rng.beta(2, 4) * 150
        carbon_intensity = rng.beta(2, 3) * 100
        disclosure_transparency = rng.beta(4, 3) * 100
        board_independence = rng.beta(5, 3) * 100
        workforce_diversity = rng.beta(3, 3) * 100
        safety_record = rng.beta(4, 2) * 100

        # ----- Ground-truth risk formulae --------------------------------
        # Environmental risk:
        #   * Industry baseline pulls strongly.
        #   * High carbon intensity pushes up risk.
        #   * Strong country EPI / disclosure pulls risk down.
        e_risk = (
            0.45 * ind["e_risk"]
            + 0.25 * carbon_intensity
            + 0.15 * c["climate_risk"]
            - 0.20 * (c["epi"] - 50)
            - 0.10 * (disclosure_transparency - 50)
        )

        # Social risk:
        #   * Industry baseline + country social progress are the
        #     dominant drivers.
        #   * Diversity and safety record dampen risk.
        s_risk = (
            0.40 * ind["s_risk"]
            - 0.20 * (c["social_progress"] - 50)
            - 0.15 * (workforce_diversity - 50)
            - 0.15 * (safety_record - 50)
            + 0.10 * (carbon_intensity - 50)  # weak cross-effect
        )

        # Governance risk:
        #   * Country rule of law and board independence dominate.
        #   * Older firms and transparent reporters get a discount.
        g_risk = (
            0.35 * ind["g_risk"]
            - 0.25 * (c["rule_of_law"] - 50)
            - 0.20 * (board_independence - 50)
            - 0.10 * (disclosure_transparency - 50)
            - 0.05 * np.clip(years_in_operation / 150, 0, 1) * 30
        )

        # Add Gaussian noise + clip to the 0-100 range.
        e_risk = float(np.clip(e_risk + rng.normal(0, 4), 0, 100))
        s_risk = float(np.clip(s_risk + rng.normal(0, 4), 0, 100))
        g_risk = float(np.clip(g_risk + rng.normal(0, 4), 0, 100))

        rows.append(pd.Series({
            "country_epi":             c["epi"],
            "country_rule_of_law":     c["rule_of_law"],
            "country_social_progress": c["social_progress"],
            "country_climate_risk":    c["climate_risk"],
            "industry_e_risk":         ind["e_risk"],
            "industry_s_risk":         ind["s_risk"],
            "industry_g_risk":         ind["g_risk"],
            "company_size":            company_size,
            "years_in_operation":      years_in_operation,
            "carbon_intensity":        carbon_intensity,
            "disclosure_transparency": disclosure_transparency,
            "board_independence":      board_independence,
            "workforce_diversity":     workforce_diversity,
            "safety_record":           safety_record,
            "e_risk":                  e_risk,
            "s_risk":                  s_risk,
            "g_risk":                  g_risk,
        }))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The model wrapper
# ---------------------------------------------------------------------------

class ESGRiskModel:
    """
    A bundle of three Random Forest regressors (one per pillar) plus
    matching SHAP explainers.

    The class is intentionally cheap to construct (~1-2 s on a laptop)
    so we can use Streamlit's ``@st.cache_resource`` to instantiate it
    once per session.
    """

    PILLARS = ("E", "S", "G")
    TARGETS = {"E": "e_risk", "S": "s_risk", "G": "g_risk"}

    def __init__(self, n_samples: int = 1500, random_state: int = 42):
        self.random_state = random_state
        self.feature_columns = FEATURE_COLUMNS

        # --- 1. Build synthetic training data ---------------------------
        df = _synthesize_training_data(n_samples=n_samples, seed=random_state)
        X = df[self.feature_columns]
        self.training_features = X  # kept for SHAP background + UI stats

        # --- 2. Fit one model + one explainer per pillar ----------------
        self.models: Dict[str, RandomForestRegressor] = {}
        self.explainers: Dict[str, shap.TreeExplainer] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}

        for pillar in self.PILLARS:
            y = df[self.TARGETS[pillar]]
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=random_state
            )

            # Random Forest is a sensible default for tabular data:
            #   * captures non-linear effects out of the box,
            #   * supports SHAP via TreeExplainer,
            #   * stable enough for a 1.5k-row demo dataset.
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=3,
                random_state=random_state,
                n_jobs=-1,
            )
            model.fit(X_tr, y_tr)

            # Track simple hold-out metrics so the UI can be honest about
            # the model's quality.
            preds = model.predict(X_te)
            self.metrics[pillar] = {
                "r2": float(r2_score(y_te, preds)),
                "mae": float(mean_absolute_error(y_te, preds)),
            }

            # TreeExplainer is exact and fast for tree ensembles.
            self.models[pillar] = model
            self.explainers[pillar] = shap.TreeExplainer(model)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, org_name: str, country: str, industry: str) -> ESGPrediction:
        """
        Run a full prediction + explanation for one organisation.
        """
        feature_row = generate_company_features(org_name, country, industry)

        pillar_preds: Dict[str, PillarPrediction] = {}
        for pillar in self.PILLARS:
            model = self.models[pillar]
            explainer = self.explainers[pillar]

            score = float(model.predict(feature_row)[0])
            score = float(np.clip(score, 0, 100))

            # SHAP values for this single row.  TreeExplainer returns
            # an array shaped (1, n_features) for single-output models.
            shap_values = explainer.shap_values(feature_row)
            if isinstance(shap_values, list):
                # Some SHAP versions wrap single-output in a list.
                shap_values = shap_values[0]
            shap_row = np.asarray(shap_values).reshape(-1)

            pillar_preds[pillar] = PillarPrediction(
                score=score,
                label=classify_risk(score),
                base_value=float(np.asarray(explainer.expected_value).reshape(-1)[0]),
                shap_values=dict(zip(self.feature_columns, shap_row.tolist())),
            )

        # Roll up to overall ESG risk using the pillar weights.
        overall = (
            PILLAR_WEIGHTS["E"] * pillar_preds["E"].score
            + PILLAR_WEIGHTS["S"] * pillar_preds["S"].score
            + PILLAR_WEIGHTS["G"] * pillar_preds["G"].score
        )

        return ESGPrediction(
            environmental=pillar_preds["E"],
            social=pillar_preds["S"],
            governance=pillar_preds["G"],
            overall_score=float(overall),
            overall_label=classify_risk(overall),
            feature_row=feature_row,
        )

    # ------------------------------------------------------------------
    # Helpers used by the Streamlit UI
    # ------------------------------------------------------------------

    def feature_importance(self, pillar: str) -> pd.DataFrame:
        """
        Global feature importance (mean absolute SHAP across the
        training set) for the requested pillar.

        We use SHAP rather than ``feature_importances_`` because SHAP
        importance is consistent and reflects directional impact on the
        model's predictions.
        """
        explainer = self.explainers[pillar]
        # Sample a subset of the training data to keep this fast.
        sample = self.training_features.sample(
            n=min(200, len(self.training_features)),
            random_state=self.random_state,
        )
        shap_matrix = explainer.shap_values(sample)
        if isinstance(shap_matrix, list):
            shap_matrix = shap_matrix[0]
        importance = np.abs(shap_matrix).mean(axis=0)
        return (
            pd.DataFrame({
                "feature": self.feature_columns,
                "importance": importance,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
