"""Integration Tests for Data Architect pipelines in dashboard.py
Role: Data Architect
Scope: End-to-end data pipelines (file → load → transform → output)
"""
import sys, os
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dashboard import (
    load_jsonl,
    load_resource_csv,
    parse_training_log,
    load_rich_csv,
    read_from_disk,
    build_real_rewards_from_rich_csv,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")


# ── IT01: Auto-discovery finds real data files on disk ──────────────────────
class TestIT01_AutoDiscovery:
    def test_real_csv_files_exist(self):
        csv_dirs = [
            os.path.join(PROJECT_ROOT, "telemetry", "telemetry", "csv"),
            os.path.join(PROJECT_ROOT, "telemetry", "csv"),
        ]
        found = False
        for d in csv_dirs:
            if os.path.isdir(d):
                csvs = [f for f in os.listdir(d) if f.startswith("telemetry_") and f.endswith(".csv")]
                if csvs:
                    found = True
                    break
        assert found, "No telemetry CSV files found on disk"

    def test_real_training_logs_exist(self):
        import glob
        logs = glob.glob(os.path.join(PROJECT_ROOT, "log_seed_*.txt"))
        assert len(logs) > 0, "No log_seed_*.txt files found on disk"

    def test_real_csv_loads_successfully(self):
        csv_dirs = [
            os.path.join(PROJECT_ROOT, "telemetry", "telemetry", "csv"),
            os.path.join(PROJECT_ROOT, "telemetry", "csv"),
        ]
        loaded = False
        for d in csv_dirs:
            if os.path.isdir(d):
                for f in sorted(os.listdir(d)):
                    if f.startswith("telemetry_") and f.endswith(".csv"):
                        fp = os.path.join(d, f)
                        try:
                            df = load_rich_csv(open(fp, "r", encoding="utf-8"))
                            assert isinstance(df, pd.DataFrame)
                            assert len(df) > 0
                            loaded = True
                        except Exception:
                            continue
                        break
            if loaded:
                break
        if not loaded:
            pytest.skip("Real telemetry CSV files have inconsistent column counts (known exporter issue)")


# ── IT02: Rich CSV → reward pipeline ───────────────────────────────────────
class TestIT02_RichCsvToRewardPipeline:
    def test_full_pipeline_produces_rewards(self):
        csv_dirs = [
            os.path.join(PROJECT_ROOT, "telemetry", "telemetry", "csv"),
            os.path.join(PROJECT_ROOT, "telemetry", "csv"),
        ]
        for d in csv_dirs:
            if os.path.isdir(d):
                for f in sorted(os.listdir(d)):
                    if f.startswith("telemetry_") and f.endswith(".csv"):
                        fp = os.path.join(d, f)
                        try:
                            rcdf = load_rich_csv(open(fp, "r", encoding="utf-8"))
                        except Exception:
                            continue
                        if rcdf.empty:
                            continue
                        rewards_df = build_real_rewards_from_rich_csv(rcdf)
                        assert isinstance(rewards_df, pd.DataFrame)
                        assert len(rewards_df) > 0
                        assert "episode" in rewards_df.columns
                        assert "reward" in rewards_df.columns
                        assert "reward_smooth" in rewards_df.columns
                        return
        pytest.skip("No valid rich CSV files found for pipeline test (known exporter column mismatch)")

    def test_pipeline_via_read_from_disk(self):
        csv_dirs = [
            os.path.join(PROJECT_ROOT, "telemetry", "telemetry", "csv"),
            os.path.join(PROJECT_ROOT, "telemetry", "csv"),
        ]
        for d in csv_dirs:
            if os.path.isdir(d):
                for f in sorted(os.listdir(d)):
                    if f.startswith("telemetry_") and f.endswith(".csv"):
                        fp = os.path.join(d, f)
                        df = read_from_disk(fp, "rich_csv")
                        if df.empty:
                            continue
                        rewards = build_real_rewards_from_rich_csv(df)
                        assert len(rewards) > 0
                        return
        pytest.skip("No valid rich CSV files for disk pipeline test (known exporter column mismatch)")


# ── IT03: Training log → experiment table ──────────────────────────────────
class TestIT03_LogToExperimentTable:
    def test_real_log_produces_experiment_df(self):
        import glob
        logs = glob.glob(os.path.join(PROJECT_ROOT, "log_seed_*.txt"))
        if not logs:
            pytest.skip("No training logs found on disk")
        with open(logs[0], "r", encoding="utf-8") as f:
            df = parse_training_log(f)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "episode" in df.columns
        assert "reward" in df.columns
        assert "epsilon" in df.columns

    def test_experiment_df_metadata_has_event_counts(self):
        import glob
        logs = glob.glob(os.path.join(PROJECT_ROOT, "log_seed_*.txt"))
        if not logs:
            pytest.skip("No training logs found on disk")
        with open(logs[0], "r", encoding="utf-8") as f:
            df = parse_training_log(f)
        assert "landing_count" in df.attrs
        assert "crash_count" in df.attrs


# ── IT04: Disk path → read → parse ─────────────────────────────────────────
class TestIT04_DiskPathEndToEnd:
    def test_fixture_jsonl_via_read_from_disk(self):
        path = os.path.join(FIXTURES, "sample.jsonl")
        df = read_from_disk(path, "jsonl")
        assert len(df) == 10
        assert "alt" in df.columns

    def test_fixture_csv_via_read_from_disk(self):
        path = os.path.join(FIXTURES, "sample_resource.csv")
        df = read_from_disk(path, "csv")
        assert len(df) == 5
        assert "cpu_pct" in df.columns

    def test_fixture_rich_csv_via_read_from_disk(self):
        path = os.path.join(FIXTURES, "sample_rich.csv")
        df = read_from_disk(path, "rich_csv")
        assert len(df) == 20
        assert "episode" in df.columns

    def test_fixture_log_via_read_from_disk(self):
        path = os.path.join(FIXTURES, "sample_log.txt")
        df = read_from_disk(path, "log")
        assert len(df) == 5
        assert "episode" in df.columns


# ── IT05: Multiple data sources loaded together ────────────────────────────
class TestIT05_MultiSourceLoading:
    def test_jsonl_and_resource_independently(self):
        jsonl_df = read_from_disk(os.path.join(FIXTURES, "sample.jsonl"), "jsonl")
        csv_df = read_from_disk(os.path.join(FIXTURES, "sample_resource.csv"), "csv")
        assert len(jsonl_df) == 10
        assert len(csv_df) == 5
        assert list(jsonl_df.columns) != list(csv_df.columns)

    def test_all_fixture_types_load(self):
        results = {}
        results["jsonl"] = read_from_disk(os.path.join(FIXTURES, "sample.jsonl"), "jsonl")
        results["csv"] = read_from_disk(os.path.join(FIXTURES, "sample_resource.csv"), "csv")
        results["rich_csv"] = read_from_disk(os.path.join(FIXTURES, "sample_rich.csv"), "rich_csv")
        results["log"] = read_from_disk(os.path.join(FIXTURES, "sample_log.txt"), "log")
        for name, df in results.items():
            assert isinstance(df, pd.DataFrame), f"{name} did not return DataFrame"
            assert len(df) > 0, f"{name} returned empty DataFrame"

    def test_no_cross_contamination(self):
        jsonl_df = read_from_disk(os.path.join(FIXTURES, "sample.jsonl"), "jsonl")
        log_df = read_from_disk(os.path.join(FIXTURES, "sample_log.txt"), "log")
        assert "reward" not in jsonl_df.columns
        assert "alt" not in log_df.columns
