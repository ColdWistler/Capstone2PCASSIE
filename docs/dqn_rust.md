# Rust DQN Extension — Technical Reference

## Overview

The `DQNRust` GDExtension class replaces the original GDScript DQN implementation with compiled Rust code via godot-rust 0.5. It runs the entire neural network forward pass, backpropagation, Adam optimization, and prioritized replay in native code with no interpreter overhead.

## Architecture

### Network Layout

```
Input (12/13)  ──►  Layer 1 (512)  ──► ReLU  ──►  Layer 2 (256)  ──► ReLU
                                                                          │
                                                            ┌─────────────┴─────────────┐
                                                            ▼                           ▼
                                                         Value (1)              Advantage (7)
                                                            │                           │
                                                            └────────── + ──────────────┘
                                                                          │
                                                                       Q(s,a) (7)
```

### Parameters

| Parameter | Value | Notes |
|---|---|---|
| Hidden layer 1 | 512 | `state_dim → 512` |
| Hidden layer 2 | 256 | `512 → 256` |
| Action dim | 7 | discrete actions |
| Learning rate | 0.001 | decays by 0.9995 per step, floor 0.0001 |
| Batch size | 64 | sampled from replay buffer |
| Replay capacity | 100,000 | prioritized SumTree |
| N-step | 3 | multi-step returns |
| Gamma | 0.99 | discount factor |
| Tau | 0.005 | Polyak averaging for target network |
| Gradient clip | 1.0 | max gradient norm |
| Epsilon | 1.0 → 0.01 | decays by 0.998 per episode |

## GDScript API

### Construction

```gdscript
var agent = DQNRust.new()
agent.init(state_dim, action_dim, hidden1, hidden2, replay_capacity, n_steps, gamma)
```

### Core Methods

| Method | Signature | Description |
|---|---|---|
| `select_action` | `(state: PackedFloat32Array) -> int` | Epsilon-greedy action selection |
| `predict_q` | `(state: PackedFloat32Array) -> PackedFloat32Array` | Q-values for all actions |
| `forward` | `(state, use_target) -> Dictionary` | Full forward pass (h1, h2, V, A, Q) |
| `push_replay` | `(state, action, reward, next_state, done)` | Store n-step transition |
| `train` | `(batch_size, gamma, grad_clip, lr) -> bool` | Sample batch, compute gradients, Adam update, Polyak target sync |
| `copy_to_target` | `()` | Hard copy online → target network |
| `polyak` | `(tau)` | Soft update target network |

### State Accessors

| Method | Returns | Description |
|---|---|---|
| `get_epsilon` / `set_epsilon` | `f64` | Exploration rate |
| `get_step_count` / `set_step_count` | `i64` | Training steps taken |
| `get_adam_step` / `set_adam_step` | `i64` | Adam step counter |
| `get_replay_size` | `i64` | Current replay buffer fill |
| `get_max_priority` / `set_max_priority` | `f64` | Max priority for new transitions |
| `get_weights_online` / `set_weights_online` | `Array[Variant]` | All online weights (8-element array) |
| `get_adam_state` / `set_adam_state` | `Array[Variant]` | All Adam optimizer state (17-element array) |

### Weight Serialization Format

`get_weights_online()` returns an array of 8 Variants:

| Index | Content |
|---|---|
| 0 | `w1` — 512×state_dim weight matrix |
| 1 | `b1` — 512 bias vector |
| 2 | `w2` — 256×512 weight matrix |
| 3 | `b2` — 256 bias vector |
| 4 | `wA` — 7×256 advantage weight matrix |
| 5 | `bA` — 7 advantage bias vector |
| 6 | `wV` — 256 value weight vector |
| 7 | `bV` — value bias scalar |

## Performance

The GDScript implementation chunked training (batch of 64 in 16-sample chunks over multiple frames) because the interpreted loop would drop frames. The Rust version processes the entire batch of 64 in a single `train()` call — the native loop finishes well before the next physics tick.

## Building

```bash
./build_rust_dqn.sh
```

Requires Rust 1.96.0+. The godot-rust 0.5 bindings include `godot-core` and `gdextension-api` static libraries (~170 MB combined in debug). First build takes ~10 minutes on an i5-10300H; incremental builds take ~2 seconds.

### Files

| Path | Purpose |
|---|---|
| `rust_dqn/src/lib.rs` | Main source — DQNRust class, forward/backward, Adam, serialization |
| `rust_dqn/src/sumtree.rs` | Prioritized replay SumTree |
| `rust_dqn/Cargo.toml` | Project manifest |
| `dqn_rust.gdextension` | Godot extension loader config |
| `build_rust_dqn.sh` | Convenience rebuild script |
