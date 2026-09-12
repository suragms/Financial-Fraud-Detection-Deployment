"""Leakage-safe feature engineering."""

from src.features.feature_engineering import (
    CATEGORICAL_FEATURES,
    EXCLUDED_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    build_features,
    engineer_features,
    extract_target,
    validate_feature_matrix,
)

__all__ = [
    "CATEGORICAL_FEATURES",
    "EXCLUDED_FEATURES",
    "MODEL_FEATURES",
    "NUMERIC_FEATURES",
    "build_features",
    "engineer_features",
    "extract_target",
    "validate_feature_matrix",
]
