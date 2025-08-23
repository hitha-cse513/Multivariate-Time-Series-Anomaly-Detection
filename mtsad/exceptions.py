from __future__ import annotations
"""Custom exceptions for the MTSAD package."""

class MTSADException(Exception):
    """Base class for MTSAD exceptions."""

class TimestampError(MTSADException):
    """Raised when timestamp column is missing or irregular."""

class DataQualityError(MTSADException):
    """Raised for data quality issues such as insufficient rows or constant features."""
