# Capstone2PCASSIE — Flight Sim DQN

Godot 4.7 flight simulation with Dueling Double DQN (D3QN) reinforcement learning agents.

## Project Overview

Simulates fixed-wing aircraft physics and provides DQN training environments. The DQN agent is implemented as a **Rust GDExtension** (`libdqn_rust.so`) for native performance, replacing the original GDScript implementation.

## Rust DQN Extension

The DQN agent lives in `rust_dqn/` and exposes the `DQNRust` class to GDScript.

| File | Purpose |
|---|---|
| `rust_dqn/src/lib.rs` | DQN agent (dueling network, Adam optimizer, prioritized replay) |
| `rust_dqn/src/sumtree.rs` | Prioritized experience replay SumTree |
| `dqn_rust.gdextension` | GDExtension entry config |
| `build_rust_dqn.sh` | Rebuild script |

### Building

```bash
./build_rust_dqn.sh
```

Requires Rust 1.96+ and the `godot` crate (v0.5). First build takes ~10 min on an i5-10300H.

### Agent Architecture

- **State**: 12 or 13 dimensions (speed, altitude, vertical speed, stall, pitch, roll, obstacle distances, fuel, battery)
- **Actions**: 7 discrete (do nothing, pitch up/down, roll left/right, throttle up/down)
- **Network**: Two hidden layers (512 -> 256 ReLU), dueling architecture (value + advantage streams from layer 2)
- **Optimizer**: Adam with bias correction and gradient clipping
- **Replay**: Prioritized experience replay (SumTree, alpha=0.6, importance sampling, 100k capacity)
- **N-step**: 3-step returns
- **Polyak averaging**: tau=0.005 for target network

## Examples

| Scene | Description |
|---|---|
| `Example1_Simple.gd` | Basic DQN training — single engine, simple rewards |
| `Example2_Complex.gd` | Advanced DQN — multiple engines, fuel/battery, obstacle avoidance |
| `Example3_Helicopter.gd` | Helicopter example (no DQN) |
| `Example4_Space.gd` | Space example (no DQN) |

### Running

```bash
godot --path .
```

Headless mode (5x speed, uncapped FPS):

```bash
godot --headless --path .
```

## Old GDScript DQN

The original GDScript DQN implementation has been fully replaced by the Rust GDExtension. Save files from the old format (SAVE_VERSION < 5) are automatically skipped and fresh training begins.

## File Reference

| File | What it does |
|---|---|
| `rust_dqn/src/lib.rs` | `DQNRust` Godot class — two-layer dueling network (512->256), Adam, prioritized replay, n-step returns, Polyak averaging. Init: `init(state_dim, action_dim, hidden1, hidden2, replay_cap, n_steps, gamma)` |
| `rust_dqn/src/sumtree.rs` | Prioritized experience replay SumTree — O(log n) weighted sampling and priority updates |
| `rust_dqn/Cargo.toml` | Rust dependencies (`godot 0.5`, `rand 0.8`) |
| `dqn_rust.gdextension` | GDExtension entry config — loads `libdqn_rust.so`, entry symbol `gdext_rust_init` |
| `example/Example1_Simple.gd` | Simple training script — single engine, 12-dim state, basic rewards |
| `example/Example2_Complex.gd` | Complex training — multiple engines, fuel/battery, obstacles, landing mode, 13-dim state |
| `build_rust_dqn.sh` | One-command rebuild (`cargo build --release`) |
| `.gitignore` | Ignores `rust_dqn/target/`, `.godot/`, `.import/`, telemetry |
| `addons/simplified_flightsim/TelemetryExporter/TelemetryExporter.gd` | Optional telemetry logger — writes flight data to `telemetry/telemetry.jsonl` |
| `project.godot` | Godot project settings — main scene `Example1_Simple.tscn` |

## Telemetry

The `TelemetryExporter` addon (under `addons/simplified_flightsim/`) writes per-frame telemetry to `telemetry/telemetry.jsonl` when configured with an `AircraftNode` path.
