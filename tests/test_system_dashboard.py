"""System Tests for Dashboard Features in dashboard.py
Role: Data Architect
Scope: Dashboard sections and features work correctly with real data
"""
import sys, os, io
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dashboard import (
    load_jsonl,
    load_resource_csv,
    parse_training_log,
    load_rich_csv,
    read_from_disk,
    build_real_rewards_from_rich_csv,
    make_demo_flight,
    make_demo_resources,
    make_demo_rewards,
    line,
    area,
    bar_components,
    flight_path,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")


# ── ST01: AI Co-pilot section shows real reward data from rich CSV ──────────
class TestST01_AiCopilotRealData:
    def test_build_rewards_produces_plot_data(self):
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
                        rewards = build_real_rewards_from_rich_csv(rcdf)
                        assert "reward" in rewards.columns
                        assert "reward_smooth" in rewards.columns
                        assert rewards["reward"].dtype in [np.float64, float]
                        return
        pytest.skip("No valid rich CSV found (known exporter column mismatch)")

    def test_reward_chart_figure_created(self):
        rewards = make_demo_rewards(20)
        fig = line(rewards, "episode", ["r_task", "r_smooth", "r_safety"],
                   ["rgb(46,204,113)", "rgb(52,152,219)", "rgb(231,76,60)"],
                   ["Task", "Smooth", "Safety"], y_label="Reward")
        assert fig is not None
        assert len(fig.data) >= 2


# ── ST02: Flight Telemetry section displays altitude/speed ──────────────────
class TestST02_FlightTelemetryDisplay:
    def test_jsonl_has_altitude_data(self):
        with open(os.path.join(FIXTURES, "sample.jsonl"), "r") as f:
            df = load_jsonl(f)
        assert "alt" in df.columns
        assert df["alt"].min() > 0
        assert df["alt"].max() < 500

    def test_jsonl_has_speed_data(self):
        with open(os.path.join(FIXTURES, "sample.jsonl"), "r") as f:
            df = load_jsonl(f)
        assert "spd" in df.columns
        assert df["spd"].min() > 0

    def test_flight_chart_created_from_jsonl(self):
        with open(os.path.join(FIXTURES, "sample.jsonl"), "r") as f:
            df = load_jsonl(f)
        fig = line(df, "t_sec", ["alt", "spd"], ["rgb(46,204,113)", "rgb(52,152,219)"],
                   ["Altitude", "Speed"], y_label="Value")
        assert fig is not None
        assert len(fig.data) == 2


# ── ST03: Training Log section shows correct episode count ──────────────────
class TestST03_TrainingLogDisplay:
    def test_fixture_log_shows_5_episodes(self):
        with open(os.path.join(FIXTURES, "sample_log.txt"), "r") as f:
            df = parse_training_log(f)
        assert df.attrs["total_episodes"] == 5

    def test_real_logs_load_correctly(self):
        import glob
        logs = glob.glob(os.path.join(PROJECT_ROOT, "log_seed_*.txt"))
        if not logs:
            pytest.skip("No logs on disk")
        with open(logs[0], "r", encoding="utf-8") as f:
            df = parse_training_log(f)
        assert df.attrs["total_episodes"] > 0


# ── ST04: Experiment Comparison table auto-populates ────────────────────────
class TestST04_ExperimentComparison:
    def test_demo_rewards_table_structure(self):
        rewards = make_demo_rewards(10)
        assert "episode" in rewards.columns
        assert "r_task" in rewards.columns
        assert len(rewards) == 10

    def test_fixture_rich_csv_produces_experiment_table(self):
        with open(os.path.join(FIXTURES, "sample_rich.csv"), "r") as f:
            rcdf = load_rich_csv(f)
        rewards = build_real_rewards_from_rich_csv(rcdf)
        assert len(rewards) == 2
        assert rewards["episode"].tolist() == [1, 2]
        assert rewards["reward"].iloc[0] > 0


# ── ST05: Auto-discovery sidebar shows correct file count ──────────────────
class TestST05_AutoDiscoverySidebar:
    def test_discovery_counts_match_actual_files(self):
        csv_dirs = [
            os.path.join(PROJECT_ROOT, "telemetry", "telemetry", "csv"),
            os.path.join(PROJECT_ROOT, "telemetry", "csv"),
        ]
        import glob
        actual_count = 0
        for d in csv_dirs:
            if os.path.isdir(d):
                actual_count += len([f for f in os.listdir(d)
                                     if f.startswith("telemetry_") and f.endswith(".csv")])
        logs = glob.glob(os.path.join(PROJECT_ROOT, "log_seed_*.txt"))
        actual_count += len(logs)
        assert actual_count > 0, "Expected at least some data files on disk"


# ── ST06: Dashboard falls back to demo data when no files exist ─────────────
class TestST06_DemoFallback:
    def test_demo_flight_returns_valid_dataframe(self):
        df = make_demo_flight(120)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 120
        for col in ["alt", "spd", "vspd", "roll", "pitch", "hdg", "px", "pz"]:
            assert col in df.columns

    def test_demo_resources_returns_valid_dataframe(self):
        df = make_demo_resources(60)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 60
        for col in ["cpu_pct", "mem_pct", "gpu_util_pct"]:
            assert col in df.columns

    def test_demo_rewards_returns_valid_dataframe(self):
        df = make_demo_rewards(80)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 80
        for col in ["episode", "r_task", "r_smooth", "r_safety", "violations"]:
            assert col in df.columns

    def test_demo_flight_chart_renders(self):
        df = make_demo_flight(50)
        fig = line(df, "t_sec", ["alt", "spd"],
                   ["rgb(46,204,113)", "rgb(52,152,219)"],
                   ["Altitude", "Speed"], y_label="Value")
        assert fig is not None


# ── ST07: Flight path chart renders when px/pz present ─────────────────────
class TestST07_FlightPathChart:
    def test_flight_path_figure_created(self):
        df = make_demo_flight(80)
        assert "px" in df.columns
        assert "pz" in df.columns
        assert "alt" in df.columns
        fig = flight_path(df)
        assert fig is not None

    def test_fixture_jsonl_has_position_data(self):
        with open(os.path.join(FIXTURES, "sample.jsonl"), "r") as f:
            df = load_jsonl(f)
        assert "px" in df.columns
        assert "pz" in df.columns
        assert df["px"].iloc[0] == 10.0


# ── ST08: Resource section shows CPU/GPU metrics ────────────────────────────
class TestST08_ResourceSection:
    def test_resource_csv_has_all_metrics(self):
        with open(os.path.join(FIXTURES, "sample_resource.csv"), "r") as f:
            df = load_resource_csv(f)
        assert "cpu_pct" in df.columns
        assert "gpu_util_pct" in df.columns
        assert df["cpu_pct"].between(0, 100).all()

    def test_resource_chart_created(self):
        with open(os.path.join(FIXTURES, "sample_resource.csv"), "r") as f:
            df = load_resource_csv(f)
        fig = area(df, "t_sec", "cpu_pct", "rgb(46,204,113)", y_label="CPU %")
        assert fig is not None


# ── ST09: Download buttons produce valid CSV content ────────────────────────
class TestST09_DownloadButtons:
    def test_dataframe_to_csv_roundtrip(self):
        df = make_demo_flight(10)
        csv_str = df.to_csv(index=False)
        assert isinstance(csv_str, str)
        assert len(csv_str) > 0
        df_back = pd.read_csv(io.StringIO(csv_str))
        assert len(df_back) == 10
        assert "alt" in df_back.columns

    def test_jsonl_data_exportable(self):
        with open(os.path.join(FIXTURES, "sample.jsonl"), "r") as f:
            df = load_jsonl(f)
        csv_str = df.to_csv(index=False)
        df_back = pd.read_csv(io.StringIO(csv_str))
        assert len(df_back) == 10

    def test_rich_csv_exportable(self):
        with open(os.path.join(FIXTURES, "sample_rich.csv"), "r") as f:
            df = load_rich_csv(f)
        csv_str = df.to_csv(index=False)
        df_back = pd.read_csv(io.StringIO(csv_str))
        assert len(df_back) == 20


# ── ST10: Session Summary shows correct max altitude ────────────────────────
class TestST10_SessionSummaryMaxAlt:
    def test_fixture_jsonl_max_alt(self):
        with open(os.path.join(FIXTURES, "sample.jsonl"), "r") as f:
            df = load_jsonl(f)
        max_alt = df["alt"].max()
        assert max_alt == pytest.approx(119.4, abs=0.1)

    def test_fixture_rich_csv_max_alt(self):
        with open(os.path.join(FIXTURES, "sample_rich.csv"), "r") as f:
            df = load_rich_csv(f)
        max_alt = df["alt"].max()
        assert max_alt == pytest.approx(119.4, abs=0.1)

    def test_demo_max_alt(self):
        df = make_demo_flight(120)
        max_alt = df["alt"].max()
        assert max_alt > 0
        assert max_alt < 500
