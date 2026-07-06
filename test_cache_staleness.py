import os
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import xarray as xr

from risk_data import (
    get_cache_freshness_warning,
    is_cache_mtime_recent,
    is_disk_cache_current,
)
from risk_dates import business_days_ago


NY = ZoneInfo("America/New_York")


def _dataset_through(day: str) -> xr.Dataset:
    return xr.Dataset(coords={"date": pd.to_datetime([day])})


def _cache_path_with_mtime(tmp_path: Path, timestamp_utc: datetime) -> str:
    cache_path = tmp_path / "factor_data.zarr"
    cache_path.mkdir()
    marker = cache_path / "marker"
    marker.write_text("cache")
    timestamp = timestamp_utc.timestamp()
    os.utime(marker, (timestamp, timestamp))
    return str(cache_path)


class CacheStalenessTests(unittest.TestCase):
    def test_business_days_ago_uses_reference_date(self):
        self.assertEqual(business_days_ago(1, current_date="2026-07-06"), date(2026, 7, 3))

    def test_recent_local_cache_mtime_is_current_without_probe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = _cache_path_with_mtime(
                Path(tmpdir),
                datetime(2026, 7, 6, 14, 0, tzinfo=timezone.utc),
            )

            def probe_should_not_run():
                raise AssertionError("Probe should not run when cache mtime is recent")

            self.assertTrue(
                is_disk_cache_current(
                    _dataset_through("2026-07-02"),
                    cache_path=cache_path,
                    current_time=datetime(2026, 7, 7, 9, 0, tzinfo=NY),
                    probe_latest_date_func=probe_should_not_run,
                )
            )

    def test_old_cache_with_probe_date_equal_to_dataset_date_is_current(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = _cache_path_with_mtime(
                Path(tmpdir),
                datetime(2026, 7, 3, 14, 0, tzinfo=timezone.utc),
            )

            self.assertTrue(
                is_disk_cache_current(
                    _dataset_through("2026-07-06"),
                    cache_path=cache_path,
                    current_time=datetime(2026, 7, 7, 9, 0, tzinfo=NY),
                    probe_latest_date_func=lambda: date(2026, 7, 6),
                )
            )

    def test_old_cache_with_newer_probe_date_is_stale(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = _cache_path_with_mtime(
                Path(tmpdir),
                datetime(2026, 7, 3, 14, 0, tzinfo=timezone.utc),
            )

            self.assertFalse(
                is_disk_cache_current(
                    _dataset_through("2026-07-02"),
                    cache_path=cache_path,
                    current_time=datetime(2026, 7, 7, 9, 0, tzinfo=NY),
                    probe_latest_date_func=lambda: date(2026, 7, 6),
                )
            )

    def test_probe_failure_keeps_cache_and_records_warning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = _cache_path_with_mtime(
                Path(tmpdir),
                datetime(2026, 7, 3, 14, 0, tzinfo=timezone.utc),
            )

            def failing_probe():
                raise RuntimeError("network unavailable")

            with self.assertLogs("risk_data", level="WARNING") as logs:
                is_current = is_disk_cache_current(
                    _dataset_through("2026-07-02"),
                    cache_path=cache_path,
                    current_time=datetime(2026, 7, 7, 9, 0, tzinfo=NY),
                    probe_latest_date_func=failing_probe,
                )

        self.assertTrue(is_current)
        self.assertIn("could not be checked", "\n".join(logs.output))
        self.assertIsNotNone(get_cache_freshness_warning())

    def test_cache_mtime_uses_new_york_date_near_utc_midnight(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = _cache_path_with_mtime(
                Path(tmpdir),
                datetime(2026, 7, 7, 0, 30, tzinfo=timezone.utc),
            )

            self.assertFalse(
                is_cache_mtime_recent(
                    cache_path,
                    current_time=datetime(2026, 7, 8, 9, 0, tzinfo=NY),
                    local_timezone="America/New_York",
                )
            )


if __name__ == "__main__":
    unittest.main()
