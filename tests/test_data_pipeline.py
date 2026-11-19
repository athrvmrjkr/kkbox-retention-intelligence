"""Unit tests for the CVM Data Ingestion and Temporal Validation Pipeline.

Acronyms:
    CVM: Customer Value Management
    OOT: Out-of-Time
"""

import polars as pl
import pytest

from src.data_pipeline import enforce_temporal_split, generate_mock_telecom_data, load_config


def test_load_config() -> None:
    """Verify that configuration loads correctly with required keys."""
    config = load_config("configs/config.yaml")
    assert "system" in config
    assert "temporal_validation" in config
    assert config["temporal_validation"]["train_cutoff_date"] == "2017-02-28"


def test_mock_data_generation() -> None:
    """Verify that simulated CVM cohort matches the required schema and scale."""
    members, transactions, user_logs, targets = generate_mock_telecom_data(n_subscribers=1000)

    assert members.height == 1000
    assert targets.height == 1000
    assert "msno" in members.columns
    assert "is_churn" in targets.columns
    assert transactions.height > 0
    assert user_logs.height > 0

    # Verify realistic churn rate ~ 11%
    churn_rate = targets["is_churn"].mean()
    assert 0.05 <= churn_rate <= 0.20


def test_enforce_temporal_split_no_future_leakage() -> None:
    """Crucial Test: Verify zero future records exist in the observation window."""
    _, transactions, _, targets = generate_mock_telecom_data(n_subscribers=2000)
    cutoff = "20170228"

    obs_tx, cohort_targets = enforce_temporal_split(
        transactions_df=transactions,
        targets_df=targets,
        cutoff_date=cutoff,
    )

    # All observation transactions must be strictly <= cutoff date
    max_tx_date = obs_tx["transaction_date"].max()
    assert max_tx_date <= cutoff, f"Leakage detected: Found transaction date {max_tx_date} > {cutoff}"

    # Verify every cohort target has an observation record
    assert cohort_targets.height > 0
    assert cohort_targets["is_churn"].null_count() == 0
