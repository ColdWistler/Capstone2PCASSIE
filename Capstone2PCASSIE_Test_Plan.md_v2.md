# Capstone2PCASSIE — Required Test Suite

**Project**: Capstone2PCASSIE (Rust GDExtension DQN Agent + Godot 4.7 Flight Simulator)  
**Date**: 2026-07-10  
**Based on**: COMMANDS.txt, paper/Capstone2 Tets.tex, research_paper_outline.md, IEEE 829 Test Plan Template

---

## Test Matrix Overview

| Test Level | Count | Status | Primary Artifact |
|------------|-------|--------|------------------|
| Unit Tests (Rust) | ~8 test functions | ❌ Not implemented | `cargo test` output |
| Integration Tests (Godot↔Rust FFI) | ~6 test cases | ❌ GUT not installed | GUT test report |
| System Tests (Training Reproduction) | 8 runs (4 seeds × 2 scenarios) | ✅ Logs exist | `log_seed_*.txt` |
| Acceptance Test (Greedy Evaluation) | 2 runs (Simple + Complex) | ⚠️ Needs Godot | `test_report.txt/html` |
| Ablation Studies | 40 runs (5 variants × 4 seeds × 2 scenarios) | ❌ Not run | CSV + paper tables |
| Transfer Learning | 3 runs (pretrain + finetune + baseline) | ❌ Not run | Comparison plots |
| Hardware/Performance | 3 benchmarks | ❌ Not run | Telemetry JSON/CSV |
| Figure Generation | 8 figures | ❌ Not generated | `fig_*.pdf` |

---

## 1. Unit Tests — Rust Core (`rust_dqn/`)

**Framework**: Built-in `cargo test`  
**Location**: `rust_dqn/tests/dqn_core.rs` + `#[cfg(test)]` modules in `lib.rs`/`sumtree.rs`  
**Run Command**: `cd rust_dqn && cargo test`

| Test ID | Test Case | Type | Priority | Target Module |
|---------|-----------|------|----------|---------------|
| UT-01 | SumTree: priority add/update/get | Functional | High | `sumtree.rs` |
| UT-02 | SumTree: sample proportional to priority | Functional | High | `sumtree.rs` |
| UT-03 | SumTree: min priority tracking | Functional | Medium | `sumtree.rs` |
| UT-04 | DuelingNetwork: forward pass shape | Functional | High | `lib.rs` |
| UT-05 | DuelingNetwork: value + advantage stream separation | Functional | High | `lib.rs` |
| UT-06 | ReplayBuffer: push/sample capacity | Functional | High | `lib.rs` |
| UT-07 | DQNAgent: action selection (ε-greedy) | Functional | High | `lib.rs` |
| UT-08 | DQNAgent: target network soft update | Functional | Medium | `lib.rs` |
| UT-09 | Adam optimizer: step updates weights | Functional | Medium | `lib.rs` |
| UT-10 | Checkpoint: save/load model round-trip | Functional | High | `lib.rs` |

---

## 2. Integration Tests — Godot ↔ Rust FFI

**Framework**: GUT (Godot Unit Test)  
**Location**: `res://tests/test_dqn_agent.gd`  
**Run Command**: `godot --headless -s res://addons/gut/gut_cmdln.gd`

| Test ID | Test Case | Type | Priority | FFI Function |
|---------|-----------|------|----------|--------------|
| IT-01 | Agent initialization with valid config | Functional | High | `init()` |
| IT-02 | Agent rejects invalid state dimension | Negative | Medium | `init()` |
| IT-03 | `select_action` returns valid action index | Functional | High | `select_action()` |
| IT-04 | `store_transition` accepts valid experience | Functional | High | `store_transition()` |
| IT-05 | `train_step` returns loss value after warmup | Functional | High | `train_step()` |
| IT-06 | `save_model`/`load_model` round-trip preserves weights | Functional | High | `save_model()`/`load_model()` |
| IT-07 | Example1_Simple scene loads without errors | Integration | High | Scene + FFI |
| IT-08 | Example2_Complex scene loads without errors | Integration | High | Scene + FFI |
| IT-09 | State extraction matches spec (12/16 dims) | Contract | High | GDScript → Rust |
| IT-10 | Reward signal in expected range [-100, 100] | Contract | Medium | GDScript → Rust |

---

## 3. System Tests — Training Reproduction (Completed)

**Framework**: Native Godot training loops  
**Logs Location**: `/home/soggy/Documents/GitHub/Capstone2PCASSIE/`  
**Status**: ✅ **COMPLETED** — 4 seeds × 2 scenarios = 8 runs

| Test ID | Scenario | Seed | Episodes | Log File | Status |
|---------|----------|------|----------|----------|--------|
| ST-01 | Example1_Simple | 42 | 200 | `log_seed_42.txt` | ✅ Done |
| ST-02 | Example1_Simple | 123 | 200 | `log_seed_123.txt` | ✅ Done |
| ST-03 | Example1_Simple | 456 | 200 | `log_seed_456.txt` | ✅ Done |
| ST-04 | Example1_Simple | 789 | 200 | `log_seed_789.txt` | ✅ Done |
| ST-05 | Example2_Complex | 42 | 200 | `log_seed_42_complex.txt` | ✅ Done |
| ST-06 | Example2_Complex | 123 | 200 | `log_seed_123_complex.txt` | ✅ Done |
| ST-07 | Example2_Complex | 456 | 200 | `log_seed_456_complex.txt` | ✅ Done |
| ST-08 | Example2_Complex | 789 | 200 | `log_seed_789_complex.txt` | ✅ Done |

**Acceptance Criteria (from paper)**:
- Best mean reward: **4811 ± 54** (Complex, final 30 episodes)
- Airspeed stability: **±8 m/s**
- **Zero crashes** in final 30 episodes

---

## 4. Acceptance Test — Greedy Evaluation (Mandatory)

**Framework**: Godot `--test` flag (implemented in Example1/2 GDScripts)  
**Run Command**: