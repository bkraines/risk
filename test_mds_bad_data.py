import numpy as np
import pandas as pd
import xarray as xr

from risk_corr_mds import drop_mds_factors, find_bad_mds_factors, normalize_mds_dates


def _corr_data_array(values, dates=None, factors=None):
    if dates is None:
        dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    if factors is None:
        factors = ["SPY", "TLT", "SHORT"]

    return xr.DataArray(
        np.array(values, dtype=float),
        dims=["corr_type", "date", "factor_name", "factor_name_1"],
        coords={
            "corr_type": [63],
            "date": dates,
            "factor_name": factors,
            "factor_name_1": factors,
        },
        name="corr",
    )


def test_find_bad_mds_factors_scans_only_relevant_dates():
    values = np.ones((1, 3, 3, 3))
    values[0, 0, 2, :] = np.nan
    values[0, 0, :, 2] = np.nan

    corr = _corr_data_array(values)

    dropped = find_bad_mds_factors(corr, dates=pd.to_datetime(["2024-01-02", "2024-01-03"]), corr_type=63)

    assert dropped.empty


def test_find_bad_mds_factors_reports_factor_and_date_count():
    values = np.ones((1, 3, 3, 3))
    values[0, 1, 2, :] = np.nan
    values[0, 1, :, 2] = np.nan
    values[0, 2, 2, :] = np.inf
    values[0, 2, :, 2] = np.inf

    corr = _corr_data_array(values)

    dropped = find_bad_mds_factors(corr, dates=corr.date.values, corr_type=63)

    short = dropped.set_index("factor_name").loc["SHORT"]
    assert short["first_bad_date"] == "2024-01-02"
    assert short["bad_date_count"] == 2
    assert short["reason"] == "missing_or_nonfinite_correlation"


def test_drop_mds_factors_removes_factor_from_both_correlation_dimensions():
    corr = _corr_data_array(np.ones((1, 1, 3, 3)), dates=pd.to_datetime(["2024-01-01"]))

    filtered = drop_mds_factors(corr, ["SHORT"])

    assert list(filtered.factor_name.values) == ["SPY", "TLT"]
    assert list(filtered.factor_name_1.values) == ["SPY", "TLT"]


def test_normalize_mds_dates_converts_static_dashboard_string_dates():
    dates = normalize_mds_dates(["2024-01-01", "2024-01-03"])

    assert dates.dtype.kind == "M"
    assert dates.max() == np.datetime64("2024-01-03")
