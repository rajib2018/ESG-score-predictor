"""
data_sources.py
================

Static reference data and feature-engineering helpers for the ESG Risk Predictor.

This module is the single source of truth for:

1. ``COUNTRY_INDICATORS`` - country-level macro indicators that influence
   the ESG risk profile of any organisation operating there (environmental
   regulation strength, rule of law, social progress, climate exposure).

2. ``INDUSTRY_BENCHMARKS`` - average per-pillar (E, S, G) risk scores for
   each industry, used both as a model feature *and* as the benchmark we
   compare a prediction against to flag Low / Medium / High risk.

3. ``RISK_THRESHOLDS`` - the global cut-offs that turn a numeric ESG risk
   score into the categorical "Low / Medium / High" label.

4. ``generate_company_features`` - deterministic feature generator that
   converts the user-facing inputs (organisation name, country, industry)
   into the rich feature vector consumed by the ML model.

All numeric values are illustrative defaults inspired by public indices
(EPI, World Bank Rule of Law, Social Progress Index, ND-GAIN climate
risk).  They are *not* a real data feed - the goal of this app is to
demonstrate the modelling and explainability pipeline.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Country-level ESG indicators
# ---------------------------------------------------------------------------
# Each indicator is normalised to a 0-100 scale where:
#   * epi              - Environmental Performance Index (higher = greener)
#   * rule_of_law      - World Bank Rule of Law (higher = stronger institutions)
#   * social_progress  - Social Progress Index (higher = better social outcomes)
#   * climate_risk     - ND-GAIN-style climate vulnerability (higher = MORE risk)
#
# These four numbers are blended into the per-pillar risk later in
# ``generate_company_features``.  The model learns from them as features,
# so changing the country directly shifts the model's prediction.
COUNTRY_INDICATORS: Dict[str, Dict[str, float]] = {
    "Switzerland":   {"epi": 87, "rule_of_law": 95, "social_progress": 90, "climate_risk": 18},
    "Sweden":        {"epi": 86, "rule_of_law": 96, "social_progress": 92, "climate_risk": 20},
    "Denmark":       {"epi": 85, "rule_of_law": 97, "social_progress": 91, "climate_risk": 22},
    "Norway":        {"epi": 84, "rule_of_law": 96, "social_progress": 92, "climate_risk": 19},
    "Finland":       {"epi": 83, "rule_of_law": 96, "social_progress": 92, "climate_risk": 21},
    "Germany":       {"epi": 78, "rule_of_law": 88, "social_progress": 86, "climate_risk": 28},
    "France":        {"epi": 77, "rule_of_law": 86, "social_progress": 85, "climate_risk": 30},
    "United Kingdom":{"epi": 78, "rule_of_law": 88, "social_progress": 85, "climate_risk": 27},
    "Netherlands":   {"epi": 79, "rule_of_law": 90, "social_progress": 88, "climate_risk": 29},
    "Japan":         {"epi": 75, "rule_of_law": 88, "social_progress": 84, "climate_risk": 35},
    "South Korea":   {"epi": 70, "rule_of_law": 82, "social_progress": 80, "climate_risk": 38},
    "Australia":     {"epi": 73, "rule_of_law": 91, "social_progress": 86, "climate_risk": 40},
    "Canada":        {"epi": 75, "rule_of_law": 92, "social_progress": 88, "climate_risk": 32},
    "United States": {"epi": 70, "rule_of_law": 84, "social_progress": 82, "climate_risk": 42},
    "Spain":         {"epi": 72, "rule_of_law": 80, "social_progress": 83, "climate_risk": 38},
    "Italy":         {"epi": 71, "rule_of_law": 76, "social_progress": 82, "climate_risk": 40},
    "Singapore":     {"epi": 76, "rule_of_law": 92, "social_progress": 85, "climate_risk": 36},
    "United Arab Emirates":
                     {"epi": 60, "rule_of_law": 75, "social_progress": 70, "climate_risk": 55},
    "Saudi Arabia":  {"epi": 55, "rule_of_law": 65, "social_progress": 60, "climate_risk": 60},
    "China":         {"epi": 55, "rule_of_law": 60, "social_progress": 65, "climate_risk": 55},
    "India":         {"epi": 45, "rule_of_law": 58, "social_progress": 55, "climate_risk": 70},
    "Brazil":        {"epi": 58, "rule_of_law": 55, "social_progress": 65, "climate_risk": 60},
    "Mexico":        {"epi": 55, "rule_of_law": 50, "social_progress": 65, "climate_risk": 58},
    "Argentina":     {"epi": 60, "rule_of_law": 52, "social_progress": 70, "climate_risk": 55},
    "South Africa":  {"epi": 50, "rule_of_law": 55, "social_progress": 60, "climate_risk": 65},
    "Nigeria":       {"epi": 35, "rule_of_law": 35, "social_progress": 45, "climate_risk": 80},
    "Russia":        {"epi": 50, "rule_of_law": 40, "social_progress": 60, "climate_risk": 60},
    "Turkey":        {"epi": 55, "rule_of_law": 50, "social_progress": 65, "climate_risk": 55},
    "Indonesia":     {"epi": 45, "rule_of_law": 50, "social_progress": 58, "climate_risk": 65},
    "Vietnam":       {"epi": 50, "rule_of_law": 55, "social_progress": 60, "climate_risk": 70},
}


# ---------------------------------------------------------------------------
# Industry-level ESG risk benchmarks (peer averages)
# ---------------------------------------------------------------------------
# These are the **expected** risk levels for a typical company in the
# industry, where higher = more risky.  They are used in two ways:
#
#   (a) As input features to the model so it learns industry context.
#   (b) As benchmarks shown to the user, so a prediction of 55 can be
#       called "Low risk" in Oil & Gas but "High risk" in Technology.
INDUSTRY_BENCHMARKS: Dict[str, Dict[str, float]] = {
    "Oil & Gas":            {"e_risk": 78, "s_risk": 55, "g_risk": 50, "esg_risk": 64},
    "Mining & Metals":      {"e_risk": 75, "s_risk": 60, "g_risk": 55, "esg_risk": 65},
    "Utilities":            {"e_risk": 70, "s_risk": 45, "g_risk": 40, "esg_risk": 55},
    "Chemicals":            {"e_risk": 70, "s_risk": 50, "g_risk": 45, "esg_risk": 57},
    "Automotive":           {"e_risk": 60, "s_risk": 45, "g_risk": 40, "esg_risk": 50},
    "Manufacturing":        {"e_risk": 55, "s_risk": 45, "g_risk": 40, "esg_risk": 47},
    "Transportation":       {"e_risk": 60, "s_risk": 45, "g_risk": 42, "esg_risk": 50},
    "Agriculture":          {"e_risk": 65, "s_risk": 55, "g_risk": 50, "esg_risk": 58},
    "Construction":         {"e_risk": 55, "s_risk": 55, "g_risk": 50, "esg_risk": 53},
    "Real Estate":          {"e_risk": 45, "s_risk": 40, "g_risk": 40, "esg_risk": 42},
    "Consumer Goods":       {"e_risk": 45, "s_risk": 45, "g_risk": 40, "esg_risk": 43},
    "Retail":               {"e_risk": 40, "s_risk": 50, "g_risk": 40, "esg_risk": 43},
    "Pharmaceuticals":      {"e_risk": 40, "s_risk": 55, "g_risk": 45, "esg_risk": 47},
    "Healthcare":           {"e_risk": 35, "s_risk": 50, "g_risk": 40, "esg_risk": 42},
    "Financial Services":   {"e_risk": 25, "s_risk": 45, "g_risk": 55, "esg_risk": 42},
    "Insurance":            {"e_risk": 25, "s_risk": 40, "g_risk": 50, "esg_risk": 38},
    "Telecommunications":   {"e_risk": 35, "s_risk": 40, "g_risk": 40, "esg_risk": 38},
    "Technology":           {"e_risk": 30, "s_risk": 45, "g_risk": 40, "esg_risk": 38},
    "Media & Entertainment":{"e_risk": 25, "s_risk": 45, "g_risk": 40, "esg_risk": 37},
    "Education":            {"e_risk": 25, "s_risk": 35, "g_risk": 35, "esg_risk": 32},
}


# ---------------------------------------------------------------------------
# Risk-level thresholds (applied identically to E, S, G and overall ESG)
# ---------------------------------------------------------------------------
# These cut-offs follow the common Sustainalytics-style buckets:
#   0  - 30  : Low risk
#   30 - 60  : Medium risk
#   60 - 100 : High risk
RISK_THRESHOLDS = {"low_max": 30.0, "medium_max": 60.0}


# ---------------------------------------------------------------------------
# The full model feature set (kept here so model.py stays in sync)
# ---------------------------------------------------------------------------
FEATURE_COLUMNS: List[str] = [
    # Country-level features
    "country_epi",
    "country_rule_of_law",
    "country_social_progress",
    "country_climate_risk",
    # Industry-level features (peer benchmarks)
    "industry_e_risk",
    "industry_s_risk",
    "industry_g_risk",
    # Company-level synthetic features (derived from organisation name)
    "company_size",            # 0-100 proxy for revenue / headcount
    "years_in_operation",      # 0-150 - older firms tend to be more transparent
    "carbon_intensity",        # 0-100 (higher = more emissions per unit revenue)
    "disclosure_transparency", # 0-100 (higher = better ESG reporting)
    "board_independence",      # 0-100 (higher = stronger governance)
    "workforce_diversity",     # 0-100 (higher = more diverse workforce)
    "safety_record",           # 0-100 (higher = stronger safety program)
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _stable_seed(text: str) -> int:
    """
    Convert an arbitrary string (e.g. an organisation name) to a stable
    32-bit integer seed.  Using ``hashlib`` instead of Python's built-in
    ``hash`` guarantees the same input always produces the same seed
    across runs and processes - critical so the same organisation always
    gets the same prediction.
    """
    digest = hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _company_synthetic_traits(org_name: str) -> Dict[str, float]:
    """
    Generate deterministic, organisation-specific traits.

    In a real deployment these would be pulled from a data warehouse
    (sustainability disclosures, financial filings, employee surveys,
    etc.).  For this demo we synthesise them from a hash of the
    organisation name so:

      * The same name always yields the same prediction.
      * Different names produce realistically distributed values.
    """
    rng = np.random.default_rng(_stable_seed(org_name))

    return {
        # Beta distributions give nicely bounded 0-100 values with
        # different shapes per trait.
        "company_size":            float(rng.beta(2, 5) * 100),
        "years_in_operation":      float(rng.beta(2, 4) * 150),
        "carbon_intensity":        float(rng.beta(2, 3) * 100),
        "disclosure_transparency": float(rng.beta(4, 3) * 100),
        "board_independence":      float(rng.beta(5, 3) * 100),
        "workforce_diversity":     float(rng.beta(3, 3) * 100),
        "safety_record":           float(rng.beta(4, 2) * 100),
    }


def generate_company_features(
    org_name: str,
    country: str,
    industry: str,
) -> pd.DataFrame:
    """
    Build the full feature row consumed by the model.

    The row is a 1-by-N DataFrame (rather than a numpy array) so that
    SHAP and the model see meaningful column names - this is what makes
    explainability readable later in the UI.
    """
    if country not in COUNTRY_INDICATORS:
        raise ValueError(f"Unknown country: {country}")
    if industry not in INDUSTRY_BENCHMARKS:
        raise ValueError(f"Unknown industry: {industry}")

    country_row = COUNTRY_INDICATORS[country]
    industry_row = INDUSTRY_BENCHMARKS[industry]
    company_row = _company_synthetic_traits(org_name)

    features = {
        "country_epi":              country_row["epi"],
        "country_rule_of_law":      country_row["rule_of_law"],
        "country_social_progress":  country_row["social_progress"],
        "country_climate_risk":     country_row["climate_risk"],
        "industry_e_risk":          industry_row["e_risk"],
        "industry_s_risk":          industry_row["s_risk"],
        "industry_g_risk":          industry_row["g_risk"],
        **company_row,
    }

    # Return as a DataFrame in the canonical column order so downstream
    # code (model + SHAP) always sees the same schema.
    return pd.DataFrame([features], columns=FEATURE_COLUMNS)


def classify_risk(score: float) -> str:
    """
    Convert a numeric risk score to a categorical label.

    Higher scores mean *higher* risk, matching the convention used by
    most ESG risk-rating providers (e.g. Sustainalytics).
    """
    if score < RISK_THRESHOLDS["low_max"]:
        return "Low"
    if score < RISK_THRESHOLDS["medium_max"]:
        return "Medium"
    return "High"


def risk_color(label: str) -> str:
    """Map a risk label to a colour used consistently across the UI."""
    return {"Low": "#2ecc71", "Medium": "#f1c40f", "High": "#e74c3c"}.get(label, "#7f8c8d")


def list_countries() -> List[str]:
    """Sorted list of supported countries (used to populate the dropdown)."""
    return sorted(COUNTRY_INDICATORS.keys())


def list_industries() -> List[str]:
    """Sorted list of supported industries (used to populate the dropdown)."""
    return sorted(INDUSTRY_BENCHMARKS.keys())
