# Changelog

## GDScript → Rust DQN Port

### What Changed

**New: Rust GDExtension (`rust_dqn/`)**
- Entire DQN agent rewritten in Rust: dueling network, Adam optimizer, prioritized replay SumTree, n-step returns, Polyak averaging
- Exposed as `DQNRust` Godot class via godot-rust 0.5 bindings
- Batch training processes all 64 samples in one native call (no chunking needed)

**GDScript side (`example/Example1_Simple.gd`, `example/Example2_Complex.gd`)**
- Removed ~500 lines of GDScript DQN code per file (network forward, backward, train_step, SumTree, Adam, etc.)
- Replaced with 6-8 calls to `agent.*` methods
- Architecture increased from single 256-hidden layer to **two layers (512 → 256 → dueling)**
- Replay buffer capacity increased from 50k to **100k**
- Save format updated (SAVE_VERSION 5), old saves skipped automatically

**Other changes**
- `dqn_rust.gdextension` — new GDExtension loader config
- `.gitignore` — ignores `rust_dqn/target/` (Rust build artifacts)
- `build_rust_dqn.sh` — one-command rebuild script
- `addons/simplified_flightsim/TelemetryExporter/TelemetryExporter.gd` — error downgraded to info when AircraftNode not set
- `README.md` — rewritten with full project docs

### What Was Removed

- All inline GDScript DQN functions: `_init_network`, `_init_adam`, `_copy_to_target`, `forward`, `predict_q`, `select_action`, `push_replay`, `sample_batch`, `train_step`, `_apply_adam`, `_polyak`
- GDScript `SumTree` class (replaced by Rust `SumTree` with identical semantics)
- Large weight/optimizer variable declarations (replaced by agent-internal state)
- `CHUNK_SIZE` splitting logic (not needed with native performance)

### Performance Impact

| Metric | Before (GDScript) | After (Rust) |
|---|---|---|
| Train batch process | 16 samples/frame (4 frames) | 64 samples/frame (1 frame) |
| Hidden layers | 1 × 256 | 2 × (512 → 256) |
| Replay capacity | 50,000 | 100,000 |
| Episode 1 reward | ~18 | ~140 |
| Episode 2 reward | ~240 | ~226 |
| FPS during training | Drops with chunking | Stable |

*Reward values are approximate from headless runs; training is stochastic.*

### Known Issues

- Old save files (SAVE_VERSION < 5) are silently skipped — training starts fresh
- Save format includes network topology dimensions; restoring weights requires matching `hidden1`/`hidden2` values
- `rust_dqn/target/` contains large Rust compilation artifacts (~170 MB) — excluded from git via `.gitignore`
