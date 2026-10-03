"""Point-in-time volatility features for dashboards and downstream models.

Entry points
------------
build_feature_table  – assemble the (ticker, date) table from the stores
load_feature_table   – read the saved table
surface_features     – implied-vol features for one surface
realized_features    – return / realised-vol features for one ticker
"""

from src.features.realized import realized_features
from src.features.surface_features import surface_features
from src.features.table import (
    FEATURE_VERSION,
    build_feature_table,
    load_feature_table,
    save_feature_table,
)

__all__ = [
    "FEATURE_VERSION",
    "build_feature_table",
    "load_feature_table",
    "realized_features",
    "save_feature_table",
    "surface_features",
]
