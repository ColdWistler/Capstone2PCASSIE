"""Unit Tests for Data Architect functions in dashboard.py
Role: Data Architect
Scope: Data loading, parsing, validation, and transformation functions
"""
import sys, os, io, json
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dashboard import (
    load_jsonl,
    load_resource_csv,
    make_demo_flight,
    make_demo_resources,
    make_demo_rewards,
    parse_training_log,
    load_rich_csv,
    read_from_disk,
    build_real_rewards_from_rich_csv,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


# ── UT01: load_jsonl – valid JSONL with all expected fields ────────────────
class TestUT01_LoadJsonlValid:
    def test_returns_dataframe_with_expected_columns(self):
        with open(os.path.join(FIXTURES, "sample.jsonl"), "r") as f:
            df = load_jsonl(f)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        for col in ["t", "alt", "spd", "vspd", "roll", "pitch", "hdg",
                     "px", "py", "pz", "g", "load", "fuel", "epow",
                     "flap", "gear", "stall", "eact"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_computes_t_sec_relative_time(self):
        with open(os.path.join(FIXTURES, "sample.jsonl"), "r") as f:
            df = load_jsonl(f)
        assert "t_sec" in df.columns
        assert df["t_sec"].iloc[0] == 0.0
        assert df["t_sec"].iloc[1] == 1.0

    def test_computes_ground_speed(self):
        with open(os.path.join(FIXTURES, "sample.jsonl"), "r") as f:
            df = load_jsonl(f)
        assert "ground_speed" in df.columns
        assert df["ground_speed"].dtype in [np.float64, np.float32, float]


# ── UT02: load_jsonl – empty file returns empty DataFrame ──────────────────
class TestUT02_LoadJsonlEmpty:
    def test_empty_file_returns_empty_df(self):
        with open(os.path.join(FIXTURES, "sample_empty.jsonl"), "r") as f:
            df = load_jsonl(f)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_empty_file_has_no_columns(self):
        with open(os.path.join(FIXTURES, "sample_empty.jsonl"), "r") as f:
            df = load_jsonl(f)
        assert len(df.columns) == 0


# ── UT03: load_jsonl – malformed JSON lines are skipped ────────────────────
class TestUT03_LoadJsonlMalformed:
    def test_skips_bad_lines_returns_valid_rows(self):
        with open(os.path.join(FIXTURES, "sample_malformed.jsonl"), "r") as f:
            df = load_jsonl(f)
        assert len(df) >= 3
        assert len(df) <= 4

    def test_valid_rows_have_correct_fields(self):
        with open(os.path.join(FIXTURES, "sample_malformed.jsonl"), "r") as f:
            df = load_jsonl(f)
        assert "t" in df.columns
        assert "alt" in df.columns
        assert 1000 in list(df["t"])
        assert 3000 in list(df["t"])
        assert 7000 in list(df["t"])


# ── UT04: load_jsonl – missing optional fields ─────────────────────────────
class TestUT04_LoadJsonlMissingFields:
    def test_missing_ground_speed_is_computed(self):
        data = [
            {"t": 0, "alt": 10, "spd": 5, "px": 0, "pz": 0},
            {"t": 1000, "alt": 12, "spd": 6, "px": 5, "pz": 1},
            {"t": 2000, "alt": 14, "spd": 7, "px": 10, "pz": 2},
        ]
        f = io.BytesIO(b"\n".join(json.dumps(r).encode() for r in data))
        df = load_jsonl(f)
        assert "ground_speed" in df.columns
        assert len(df) == 3

    def test_missing_px_does_not_crash(self):
        data = [
            {"t": 0, "alt": 10, "spd": 5},
            {"t": 1000, "alt": 12, "spd": 6},
        ]
        f = io.BytesIO(b"\n".join(json.dumps(r).encode() for r in data))
        df = load_jsonl(f)
        assert len(df) == 2


# ── UT05: load_resource_csv – valid CSV with all columns ───────────────────
class TestUT05_LoadResourceCsvValid:
    def test_returns_all_columns(self):
        with open(os.path.join(FIXTURES, "sample_resource.csv"), "r") as f:
            df = load_resource_csv(f)
        assert len(df) == 5
        for col in ["timestamp", "cpu_pct", "mem_pct", "gpu_util_pct",
                     "gpu_mem_mb", "disk_read_mbs", "disk_write_mbs"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_adds_t_sec_column(self):
        with open(os.path.join(FIXTURES, "sample_resource.csv"), "r") as f:
            df = load_resource_csv(f)
        assert "t_sec" in df.columns
        assert list(df["t_sec"]) == [0, 1, 2, 3, 4]


# ── UT06: load_resource_csv – CSV without GPU columns ──────────────────────
class TestUT06_LoadResourceCsvNoGpu:
    def test_loads_without_gpu_columns(self):
        with open(os.path.join(FIXTURES, "sample_resource_nogpu.csv"), "r") as f:
            df = load_resource_csv(f)
        assert len(df) == 5
        assert "cpu_pct" in df.columns
        assert "gpu_util_pct" not in df.columns
        assert "gpu_mem_mb" not in df.columns
        assert "t_sec" in df.columns


# ── UT07: parse_training_log – valid log ───────────────────────────────────
class TestUT07_ParseTrainingLogValid:
    def test_parses_all_episodes(self):
        with open(os.path.join(FIXTURES, "sample_log.txt"), "r") as f:
            df = parse_training_log(f)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5

    def test_episode_columns_present(self):
        with open(os.path.join(FIXTURES, "sample_log.txt"), "r") as f:
            df = parse_training_log(f)
        for col in ["episode", "reward", "steps", "epsilon", "best_reward", "reward_smooth"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_episode_values_correct(self):
        with open(os.path.join(FIXTURES, "sample_log.txt"), "r") as f:
            df = parse_training_log(f)
        assert list(df["episode"]) == [1, 2, 3, 4, 5]
        assert df["reward"].iloc[0] == pytest.approx(47.7, abs=0.01)
        assert df["reward"].iloc[1] == pytest.approx(1984.6, abs=0.01)

    def test_metadata_attrs_set(self):
        with open(os.path.join(FIXTURES, "sample_log.txt"), "r") as f:
            df = parse_training_log(f)
        assert df.attrs["total_episodes"] == 5
        assert df.attrs["landing_count"] == 1
        assert df.attrs["crash_count"] == 1

    def test_reward_smooth_is_rolling_average(self):
        with open(os.path.join(FIXTURES, "sample_log.txt"), "r") as f:
            df = parse_training_log(f)
        assert len(df["reward_smooth"]) == 5
        assert df["reward_smooth"].iloc[0] == pytest.approx(47.7, abs=0.01)


# ── UT08: parse_training_log – empty log ───────────────────────────────────
class TestUT08_ParseTrainingLogEmpty:
    def test_empty_log_returns_empty_df(self):
        with open(os.path.join(FIXTURES, "sample_empty_log.txt"), "r") as f:
            df = parse_training_log(f)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# ── UT09: parse_training_log – mixed valid and invalid lines ────────────────
class TestUT09_ParseTrainingLogMixed:
    def test_only_valid_episode_lines_parsed(self):
        mixed_log = (
            "some preamble text\n"
            "Episode 1 | reward: 100.5 | steps: 50 | eps: 1.000 | best: 100.5\n"
            "random noise line\n"
            "Episode 2 | reward: 200.3 | steps: 80 | eps: 0.950 | best: 200.3\n"
        )
        f = io.StringIO(mixed_log)
        df = parse_training_log(f)
        assert len(df) == 2
        assert list(df["episode"]) == [1, 2]

    def test_empty_string_lines_ignored(self):
        f = io.StringIO("Episode 1 | reward: 50.0 | steps: 25 | eps: 1.000 | best: 50.0\n\n\n\n")
        df = parse_training_log(f)
        assert len(df) == 1


# ── UT10: load_rich_csv – valid 46-column CSV with DQN data ────────────────
class TestUT10_LoadRichCsvValid:
    def test_returns_dataframe_with_rows(self):
        with open(os.path.join(FIXTURES, "sample_rich.csv"), "r") as f:
            df = load_rich_csv(f)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 20

    def test_has_dqn_columns(self):
        with open(os.path.join(FIXTURES, "sample_rich.csv"), "r") as f:
            df = load_rich_csv(f)
        for col in ["state_0", "state_15", "Q_0", "Q_6", "reward", "epsilon", "episode", "step_count"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_has_telemetry_columns(self):
        with open(os.path.join(FIXTURES, "sample_rich.csv"), "r") as f:
            df = load_rich_csv(f)
        for col in ["alt", "spd", "vspd", "roll", "pitch", "hdg"]:
            assert col in df.columns, f"Missing column: {col}"


# ── UT11: load_rich_csv – column renaming ──────────────────────────────────
class TestUT11_LoadRichCsvRenaming:
    def test_renames_t_ms_to_t(self):
        with open(os.path.join(FIXTURES, "sample_rich.csv"), "r") as f:
            df = load_rich_csv(f)
        assert "t" in df.columns
        assert "t_ms" not in df.columns

    def test_renames_alt_m_to_alt(self):
        with open(os.path.join(FIXTURES, "sample_rich.csv"), "r") as f:
            df = load_rich_csv(f)
        assert "alt" in df.columns
        assert "alt_m" not in df.columns

    def test_computes_t_sec(self):
        with open(os.path.join(FIXTURES, "sample_rich.csv"), "r") as f:
            df = load_rich_csv(f)
        assert "t_sec" in df.columns
        assert df["t_sec"].iloc[0] == 0.0
        assert df["t_sec"].iloc[1] == 1.0

    def test_all_col_map_renames_applied(self):
        with open(os.path.join(FIXTURES, "sample_rich.csv"), "r") as f:
            df = load_rich_csv(f)
        for old in ["t_ms", "alt_m", "spd_ms", "fwd_spd_ms", "vspd_ms",
                     "g_force", "load_factor", "roll_deg", "pitch_deg", "hdg_deg"]:
            assert old not in df.columns, f"Column {old} should have been renamed"


# ── UT12: read_from_disk – non-existent path returns empty DataFrame ───────
class TestUT12_ReadFromDiskMissing:
    def test_nonexistent_jsonl_returns_empty(self):
        df = read_from_disk("C:\\nonexistent\\file.jsonl", "jsonl")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_nonexistent_csv_returns_empty(self):
        df = read_from_disk("C:\\nonexistent\\file.csv", "csv")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_nonexistent_log_returns_empty(self):
        df = read_from_disk("C:\\nonexistent\\file.txt", "log")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_unknown_type_returns_empty(self):
        df = read_from_disk("C:\\nonexistent\\file.xyz", "unknown")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# ── UT13: build_real_rewards_from_rich_csv – per-episode aggregation ────────
class TestUT13_BuildRewardsFromRichCsv:
    def test_produces_per_episode_rows(self):
        with open(os.path.join(FIXTURES, "sample_rich.csv"), "r") as f:
            rcdf = load_rich_csv(f)
        df = build_real_rewards_from_rich_csv(rcdf)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df["episode"]) == [1, 2]

    def test_reward_columns_present(self):
        with open(os.path.join(FIXTURES, "sample_rich.csv"), "r") as f:
            rcdf = load_rich_csv(f)
        df = build_real_rewards_from_rich_csv(rcdf)
        for col in ["episode", "reward", "best_reward", "epsilon", "steps",
                     "violations", "reward_smooth"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_step_count_matches_group_size(self):
        with open(os.path.join(FIXTURES, "sample_rich.csv"), "r") as f:
            rcdf = load_rich_csv(f)
        df = build_real_rewards_from_rich_csv(rcdf)
        assert df["steps"].iloc[0] == 10
        assert df["steps"].iloc[1] == 10

    def test_violations_count_stall(self):
        with open(os.path.join(FIXTURES, "sample_rich.csv"), "r") as f:
            rcdf = load_rich_csv(f)
        df = build_real_rewards_from_rich_csv(rcdf)
        assert df["violations"].iloc[0] == 0


# ── UT14: build_real_rewards_from_rich_csv – empty DataFrame ───────────────
class TestUT14_BuildRewardsEmpty:
    def test_empty_input_returns_empty(self):
        df = build_real_rewards_from_rich_csv(pd.DataFrame())
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_missing_episode_column_returns_empty(self):
        df_input = pd.DataFrame({"reward": [1, 2, 3]})
        df = build_real_rewards_from_rich_csv(df_input)
        assert len(df) == 0

    def test_missing_reward_column_returns_empty(self):
        df_input = pd.DataFrame({"episode": [1, 1, 2]})
        df = build_real_rewards_from_rich_csv(df_input)
        assert len(df) == 0
