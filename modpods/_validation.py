from __future__ import annotations

import pandas as pd


class ValidationError(TypeError, ValueError):
    """Raised when modpods input validation fails."""


def validate_system_data(system_data: pd.DataFrame) -> None:
    if not isinstance(system_data, pd.DataFrame):
        raise ValidationError(
            f"system_data must be a pandas DataFrame, got {type(system_data).__name__}"
        )
    if not isinstance(system_data.index, pd.DatetimeIndex):
        raise ValidationError("system_data index must be a pandas DatetimeIndex")
    if system_data.empty:
        raise ValidationError("system_data must not be empty")
    if not pd.api.types.is_numeric_dtype(system_data.values):
        raise ValidationError("system_data must contain only numeric values")


def validate_columns(system_data: pd.DataFrame, columns: list[str], name: str) -> None:
    if not isinstance(columns, list):
        raise ValidationError(
            f"{name} must be a list of strings, got {type(columns).__name__}"
        )
    if not all(isinstance(c, str) for c in columns):
        raise ValidationError(f"{name} must contain only strings")
    if not columns:
        raise ValidationError(f"{name} must not be empty")
    missing = [c for c in columns if c not in system_data.columns]
    if missing:
        raise ValidationError(
            f"{name} contains columns not in system_data: {missing}"
        )
