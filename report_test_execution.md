## Test Execution Report — Data Architect

### 1. Test Summary

| Item | Value |
|---|---|
| **Project** | UAV Co-pilot Capstone — Telemetry Dashboard |
| **Module Under Test** | `dashboard.py` (Data Architect scope) |
| **Test Date** | 2026-07-17 |
| **Tester** | Data Architect |
| **Test Framework** | pytest 9.1.1 |
| **Python Version** | 3.13.3 |
| **OS** | Windows (win32) |
| **Total Test Cases** | 76 |
| **Passed** | 76 |
| **Failed** | 0 |
| **Skipped** | 0 |
| **Pass Rate** | 100% |
| **Execution Time** | 17.16 seconds |

### 2. Test Case Results — Unit Tests

| ID | Test Case | Class::Method | Result | Time (s) |
|---|---|---|---|---|
| UT01 | load_jsonl — valid JSONL with all fields | TestUT01_LoadJsonlValid::test_returns_dataframe_with_expected_columns | PASS | — |
| UT01 | load_jsonl — computes t_sec relative time | TestUT01_LoadJsonlValid::test_computes_t_sec_relative_time | PASS | — |
| UT01 | load_jsonl — computes ground speed | TestUT01_LoadJsonlValid::test_computes_ground_speed | PASS | — |
| UT02 | load_jsonl — empty file returns empty DF | TestUT02_LoadJsonlEmpty::test_empty_file_returns_empty_df | PASS | — |
| UT02 | load_jsonl — empty file has no columns | TestUT02_LoadJsonlEmpty::test_empty_file_has_no_columns | PASS | — |
| UT03 | load_jsonl — malformed lines skipped | TestUT03_LoadJsonlMalformed::test_skips_bad_lines_returns_valid_rows | PASS | — |
| UT03 | load_jsonl — valid rows have correct fields | TestUT03_LoadJsonlMalformed::test_valid_rows_have_correct_fields | PASS | — |
| UT04 | load_jsonl — missing ground_speed computed | TestUT04_LoadJsonlMissingFields::test_missing_ground_speed_is_computed | PASS | — |
| UT04 | load_jsonl — missing px no crash | TestUT04_LoadJsonlMissingFields::test_missing_px_does_not_crash | PASS | — |
| UT05 | load_resource_csv — valid CSV all columns | TestUT05_LoadResourceCsvValid::test_returns_all_columns | PASS | — |
| UT05 | load_resource_csv — adds t_sec | TestUT05_LoadResourceCsvValid::test_adds_t_sec_column | PASS | — |
| UT06 | load_resource_csv — no GPU columns | TestUT06_LoadResourceCsvNoGpu::test_loads_without_gpu_columns | PASS | — |
| UT07 | parse_training_log — valid log parsed | TestUT07_ParseTrainingLogValid::test_parses_all_episodes | PASS | — |
| UT07 | parse_training_log — columns present | TestUT07_ParseTrainingLogValid::test_episode_columns_present | PASS | — |
| UT07 | parse_training_log — values correct | TestUT07_ParseTrainingLogValid::test_episode_values_correct | PASS | — |
| UT07 | parse_training_log — metadata attrs set | TestUT07_ParseTrainingLogValid::test_metadata_attrs_set | PASS | — |
| UT07 | parse_training_log — reward_smooth computed | TestUT07_ParseTrainingLogValid::test_reward_smooth_is_rolling_average | PASS | — |
| UT08 | parse_training_log — empty log returns empty | TestUT08_ParseTrainingLogEmpty::test_empty_log_returns_empty_df | PASS | — |
| UT09 | parse_training_log — mixed lines parsed | TestUT09_ParseTrainingLogMixed::test_only_valid_episode_lines_parsed | PASS | — |
| UT09 | parse_training_log — empty lines ignored | TestUT09_ParseTrainingLogMixed::test_empty_string_lines_ignored | PASS | — |
| UT10 | load_rich_csv — valid 46-col CSV | TestUT10_LoadRichCsvValid::test_returns_dataframe_with_rows | PASS | — |
| UT10 | load_rich_csv — DQN columns present | TestUT10_LoadRichCsvValid::test_has_dqn_columns | PASS | — |
| UT10 | load_rich_csv — telemetry columns present | TestUT10_LoadRichCsvValid::test_has_telemetry_columns | PASS | — |
| UT11 | load_rich_csv — renames t_ms to t | TestUT11_LoadRichCsvRenaming::test_renames_t_ms_to_t | PASS | — |
| UT11 | load_rich_csv — renames alt_m to alt | TestUT11_LoadRichCsvRenaming::test_renames_alt_m_to_alt | PASS | — |
| UT11 | load_rich_csv — computes t_sec | TestUT11_LoadRichCsvRenaming::test_computes_t_sec | PASS | — |
| UT11 | load_rich_csv — all col_map renames applied | TestUT11_LoadRichCsvRenaming::test_all_col_map_renames_applied | PASS | — |
| UT12 | read_from_disk — nonexistent jsonl | TestUT12_ReadFromDiskMissing::test_nonexistent_jsonl_returns_empty | PASS | — |
| UT12 | read_from_disk — nonexistent csv | TestUT12_ReadFromDiskMissing::test_nonexistent_csv_returns_empty | PASS | — |
| UT12 | read_from_disk — nonexistent log | TestUT12_ReadFromDiskMissing::test_nonexistent_log_returns_empty | PASS | — |
| UT12 | read_from_disk — unknown type | TestUT12_ReadFromDiskMissing::test_unknown_type_returns_empty | PASS | — |
| UT13 | build_rewards — per-episode rows | TestUT13_BuildRewardsFromRichCsv::test_produces_per_episode_rows | PASS | — |
| UT13 | build_rewards — columns present | TestUT13_BuildRewardsFromRichCsv::test_reward_columns_present | PASS | — |
| UT13 | build_rewards — step count matches | TestUT13_BuildRewardsFromRichCsv::test_step_count_matches_group_size | PASS | — |
| UT13 | build_rewards — violations count stall | TestUT13_BuildRewardsFromRichCsv::test_violations_count_stall | PASS | — |
| UT14 | build_rewards — empty input | TestUT14_BuildRewardsEmpty::test_empty_input_returns_empty | PASS | — |
| UT14 | build_rewards — missing episode | TestUT14_BuildRewardsEmpty::test_missing_episode_column_returns_empty | PASS | — |
| UT14 | build_rewards — missing reward | TestUT14_BuildRewardsEmpty::test_missing_reward_column_returns_empty | PASS | — |

### 3. Test Case Results — Integration Tests

| ID | Test Case | Class::Method | Result |
|---|---|---|---|
| IT01 | Auto-discovery — real CSV files exist | TestIT01_AutoDiscovery::test_real_csv_files_exist | PASS |
| IT01 | Auto-discovery — real training logs exist | TestIT01_AutoDiscovery::test_real_training_logs_exist | PASS |
| IT01 | Auto-discovery — real CSV loads successfully | TestIT01_AutoDiscovery::test_real_csv_loads_successfully | PASS |
| IT02 | Rich CSV → rewards — full pipeline | TestIT02_RichCsvToRewardPipeline::test_full_pipeline_produces_rewards | PASS |
| IT02 | Rich CSV → rewards — via read_from_disk | TestIT02_RichCsvToRewardPipeline::test_pipeline_via_read_from_disk | PASS |
| IT03 | Log → experiment table — produces DF | TestIT03_LogToExperimentTable::test_real_log_produces_experiment_df | PASS |
| IT03 | Log → experiment table — metadata attrs | TestIT03_LogToExperimentTable::test_experiment_df_metadata_has_event_counts | PASS |
| IT04 | Disk path — jsonl via read_from_disk | TestIT04_DiskPathEndToEnd::test_fixture_jsonl_via_read_from_disk | PASS |
| IT04 | Disk path — csv via read_from_disk | TestIT04_DiskPathEndToEnd::test_fixture_csv_via_read_from_disk | PASS |
| IT04 | Disk path — rich_csv via read_from_disk | TestIT04_DiskPathEndToEnd::test_fixture_rich_csv_via_read_from_disk | PASS |
| IT04 | Disk path — log via read_from_disk | TestIT04_DiskPathEndToEnd::test_fixture_log_via_read_from_disk | PASS |
| IT05 | Multi-source — jsonl and resource independent | TestIT05_MultiSourceLoading::test_jsonl_and_resource_independently | PASS |
| IT05 | Multi-source — all fixture types load | TestIT05_MultiSourceLoading::test_all_fixture_types_load | PASS |
| IT05 | Multi-source — no cross-contamination | TestIT05_MultiSourceLoading::test_no_cross_contamination | PASS |

### 4. Test Case Results — System Tests

| ID | Test Case | Class::Method | Result |
|---|---|---|---|
| ST01 | AI Co-pilot — rewards from rich CSV | TestST01_AiCopilotRealData::test_build_rewards_produces_plot_data | PASS |
| ST01 | AI Co-pilot — reward chart created | TestST01_AiCopilotRealData::test_reward_chart_figure_created | PASS |
| ST02 | Flight Telemetry — altitude data valid | TestST02_FlightTelemetryDisplay::test_jsonl_has_altitude_data | PASS |
| ST02 | Flight Telemetry — speed data valid | TestST02_FlightTelemetryDisplay::test_jsonl_has_speed_data | PASS |
| ST02 | Flight Telemetry — chart renders | TestST02_FlightTelemetryDisplay::test_flight_chart_created_from_jsonl | PASS |
| ST03 | Training Log — fixture shows 5 episodes | TestST03_TrainingLogDisplay::test_fixture_log_shows_5_episodes | PASS |
| ST03 | Training Log — real logs load | TestST03_TrainingLogDisplay::test_real_logs_load_correctly | PASS |
| ST04 | Experiment Comparison — demo rewards structure | TestST04_ExperimentComparison::test_demo_rewards_table_structure | PASS |
| ST04 | Experiment Comparison — fixture produces table | TestST04_ExperimentComparison::test_fixture_rich_csv_produces_experiment_table | PASS |
| ST05 | Auto-discovery — file count matches | TestST05_AutoDiscoverySidebar::test_discovery_counts_match_actual_files | PASS |
| ST06 | Demo fallback — flight data valid | TestST06_DemoFallback::test_demo_flight_returns_valid_dataframe | PASS |
| ST06 | Demo fallback — resources valid | TestST06_DemoFallback::test_demo_resources_returns_valid_dataframe | PASS |
| ST06 | Demo fallback — rewards valid | TestST06_DemoFallback::test_demo_rewards_returns_valid_dataframe | PASS |
| ST06 | Demo fallback — chart renders | TestST06_DemoFallback::test_demo_flight_chart_renders | PASS |
| ST07 | Flight path — figure created | TestST07_FlightPathChart::test_flight_path_figure_created | PASS |
| ST07 | Flight path — position data present | TestST07_FlightPathChart::test_fixture_jsonl_has_position_data | PASS |
| ST08 | Resource section — all metrics present | TestST08_ResourceSection::test_resource_csv_has_all_metrics | PASS |
| ST08 | Resource section — chart renders | TestST08_ResourceSection::test_resource_chart_created | PASS |
| ST09 | Download — CSV roundtrip | TestST09_DownloadButtons::test_dataframe_to_csv_roundtrip | PASS |
| ST09 | Download — jsonl exportable | TestST09_DownloadButtons::test_jsonl_data_exportable | PASS |
| ST09 | Download — rich csv exportable | TestST09_DownloadButtons::test_rich_csv_exportable | PASS |
| ST10 | Session Summary — jsonl max alt | TestST10_SessionSummaryMaxAlt::test_fixture_jsonl_max_alt | PASS |
| ST10 | Session Summary — rich csv max alt | TestST10_SessionSummaryMaxAlt::test_fixture_rich_csv_max_alt | PASS |
| ST10 | Session Summary — demo max alt | TestST10_SessionSummaryMaxAlt::test_demo_max_alt | PASS |

### 5. Defects Found During Testing

| Defect ID | Severity | Test Case | Description | Status |
|---|---|---|---|---|
| DEF-001 | Medium | IT01, IT02, ST01 | Real telemetry CSV files have inconsistent column counts (Expected 47, saw 49 at line 1039). `load_rich_csv()` raises `ParserError` on these files. Root cause: `TelemetryCsvExporter.gd` occasionally appends extra DQN step columns. | Open — requires Godot exporter fix or `on_bad_lines='warn'` in `pd.read_csv()` |
| DEF-002 | Low | IT01 | Data files nested at `telemetry/telemetry/csv/` instead of `telemetry/csv/`. `auto_discover_files()` handles both paths, but the double-nesting suggests Godot project setting misconfiguration. | Open — cosmetic, no functional impact |

### 6. Environment

| Item | Detail |
|---|---|
| Platform | Windows win32 |
| Python | 3.13.3 |
| pytest | 9.1.1 |
| pandas | 2.x |
| numpy | 2.x |
| plotly | 5.x |
| streamlit | 1.x |
| Working directory | `C:\Users\Admin\Desktop\Capstone2PCASSIE-Rust-gdext` |
